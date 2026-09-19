// Parallel Contour Construction using LuisaCompute
// Implements GPU-parallel connected components (hook-compress union-find)
// + parity propagation + consistency check for intersection contour grouping.

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ============================================================
// Host-side CSR Graph
// ============================================================

struct HostCSRGraph
{
	uint			  num_nodes = 0;
	std::vector<uint> row_ptr; // [num_nodes + 1]
	std::vector<uint> col_idx; // directed adjacency
	std::vector<uint> flip_w;  // 0/1 per directed edge
};

static HostCSRGraph make_undirected_csr(
	uint											 num_nodes,
	const std::vector<std::tuple<uint, uint, uint>>& undirected_edges)
{
	HostCSRGraph g;
	g.num_nodes = num_nodes;
	g.row_ptr.assign(num_nodes + 1, 0);

	for (const auto& [u, v, w] : undirected_edges)
	{
		g.row_ptr[u + 1]++;
		g.row_ptr[v + 1]++;
	}
	for (uint i = 0; i < num_nodes; ++i)
		g.row_ptr[i + 1] += g.row_ptr[i];

	const uint M = g.row_ptr.back();
	g.col_idx.resize(M);
	g.flip_w.resize(M);
	std::vector<uint> cursor(g.row_ptr.begin(), g.row_ptr.end());

	for (const auto& [u, v, w] : undirected_edges)
	{
		uint p0 = cursor[u]++;
		uint p1 = cursor[v]++;
		g.col_idx[p0] = v;
		g.flip_w[p0] = w & 1u;
		g.col_idx[p1] = u;
		g.flip_w[p1] = w & 1u;
	}
	return g;
}

// ============================================================
// Graph Generators
// ============================================================

// Many independent chains of chain_length nodes.
// The first num_cycles chains are closed into cycles (may introduce inconsistency).
static HostCSRGraph generate_chains(uint num_chains, uint chain_length,
	uint num_cycles, std::mt19937& rng)
{
	const uint								  N = num_chains * chain_length;
	std::vector<std::tuple<uint, uint, uint>> edges;
	std::uniform_int_distribution<uint>		  w_dist(0, 1);

	for (uint c = 0; c < num_chains; ++c)
	{
		uint base = c * chain_length;
		for (uint i = 0; i < chain_length - 1; ++i)
			edges.emplace_back(base + i, base + i + 1, w_dist(rng));
		if (c < num_cycles && chain_length >= 3)
			edges.emplace_back(base, base + chain_length - 1, w_dist(rng));
	}
	return make_undirected_csr(N, edges);
}

// 2D grid graph: rows x cols, single connected component.
static HostCSRGraph generate_grid(uint rows, uint cols, std::mt19937& rng)
{
	const uint								  N = rows * cols;
	std::vector<std::tuple<uint, uint, uint>> edges;
	std::uniform_int_distribution<uint>		  w_dist(0, 1);
	auto									  idx = [cols](uint r, uint c)
	{ return r * cols + c; };

	for (uint r = 0; r < rows; ++r)
		for (uint c = 0; c < cols; ++c)
		{
			if (c + 1 < cols)
				edges.emplace_back(idx(r, c), idx(r, c + 1), w_dist(rng));
			if (r + 1 < rows)
				edges.emplace_back(idx(r, c), idx(r + 1, c), w_dist(rng));
		}
	return make_undirected_csr(N, edges);
}

// Random sparse graph: N nodes split into num_components groups.
// Each component gets a random spanning tree + extra edges to reach avg_degree.
static HostCSRGraph generate_random_sparse(uint num_nodes, uint avg_degree,
	uint num_components, std::mt19937& rng)
{
	std::vector<std::tuple<uint, uint, uint>> edges;
	std::uniform_int_distribution<uint>		  w_dist(0, 1);

	uint comp_size = std::max(num_nodes / std::max(num_components, 1u), 1u);
	for (uint comp = 0; comp < num_components; ++comp)
	{
		uint base = comp * comp_size;
		uint end = (comp + 1 == num_components) ? num_nodes : base + comp_size;
		uint sz = end - base;
		if (sz <= 1)
			continue;

		// Random spanning tree
		std::vector<uint> perm(sz);
		std::iota(perm.begin(), perm.end(), base);
		std::shuffle(perm.begin(), perm.end(), rng);
		for (uint i = 0; i + 1 < sz; ++i)
			edges.emplace_back(std::min(perm[i], perm[i + 1]),
				std::max(perm[i], perm[i + 1]), w_dist(rng));

		// Extra random edges
		uint								extra = std::max(int(sz * avg_degree / 2) - int(sz - 1), 0);
		std::uniform_int_distribution<uint> node_dist(base, end - 1);
		for (uint i = 0; i < extra; ++i)
		{
			uint u = node_dist(rng);
			uint v = node_dist(rng);
			if (u != v)
				edges.emplace_back(std::min(u, v), std::max(u, v), w_dist(rng));
		}
	}
	return make_undirected_csr(num_nodes, edges);
}

// ============================================================
// CPU Reference (BFS flood fill)
// ============================================================

struct CPUResult
{
	std::vector<std::vector<uint>> contours;
	std::vector<uint>			   mesh_index;
	bool						   inconsistent = false;
};

static CPUResult cpu_reference(const HostCSRGraph& g)
{
	CPUResult		 out;
	const uint		 n = g.num_nodes;
	std::vector<int> visited(n, 0);
	out.mesh_index.assign(n, 0);

	for (uint s = 0; s < n; ++s)
	{
		if (visited[s])
			continue;
		std::queue<uint> q;
		q.push(s);
		visited[s] = 1;
		out.mesh_index[s] = 0;
		std::vector<uint> comp = { s };

		while (!q.empty())
		{
			uint u = q.front();
			q.pop();
			for (uint e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e)
			{
				uint v = g.col_idx[e];
				uint want = out.mesh_index[u] ^ g.flip_w[e];
				if (!visited[v])
				{
					visited[v] = 1;
					out.mesh_index[v] = want;
					q.push(v);
					comp.push_back(v);
				}
				else if (out.mesh_index[v] != want)
				{
					out.inconsistent = true;
				}
			}
		}
		std::sort(comp.begin(), comp.end());
		out.contours.push_back(comp);
	}
	std::sort(out.contours.begin(), out.contours.end());
	return out;
}

// ============================================================
// Verification: compare GPU contours & parity against CPU
// ============================================================

static bool verify_results(const HostCSRGraph& g, const CPUResult& cpu,
	const std::vector<std::vector<uint>>& gpu_contours,
	const std::vector<uint>&			  gpu_parity,
	const std::vector<uint>&			  gpu_comp_root,
	bool gpu_inconsistent, uint N)
{
	bool same_contours = (cpu.contours == gpu_contours);
	bool same_inconsistency = (cpu.inconsistent == gpu_inconsistent);

	// Per-component consistency check using CPU mesh_index.
	// In inconsistent components the parity assignment depends on traversal order,
	// so we only compare parities for components where all edges are satisfied.
	std::unordered_map<uint, bool> root_consistent;
	for (uint i = 0; i < N; ++i)
		root_consistent[gpu_comp_root[i]] = true;

	for (uint u = 0; u < N; ++u)
	{
		for (uint e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e)
		{
			uint v = g.col_idx[e];
			if (gpu_comp_root[u] != gpu_comp_root[v])
				continue;
			if ((cpu.mesh_index[u] ^ cpu.mesh_index[v]) != g.flip_w[e])
				root_consistent[gpu_comp_root[u]] = false;
		}
	}

	bool same_mesh = true;
	for (uint i = 0; i < N; ++i)
	{
		if (!root_consistent[gpu_comp_root[i]])
			continue;
		if (cpu.mesh_index[i] != gpu_parity[i])
		{
			same_mesh = false;
			break;
		}
	}

	return same_contours && same_mesh && same_inconsistency;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
	luisa::log_level_info();
	LUISA_INFO("Test Parallel Contour Construction");

	// ---- Init GPU ----
#if defined(__APPLE__)
	std::string backend = "metal";
#else
	std::string backend = "cuda";
#endif
	const std::string			 binary_path(argv[0]);
	luisa::compute::Context		 context{ binary_path };
	luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
	if (device_names.empty())
	{
		LUISA_WARNING("No hardware device found.");
		return 1;
	}
	if (argc >= 2)
		backend = argv[1];
	luisa::compute::Device device = context.create_device(backend, nullptr, true);
	luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

	using namespace luisa::compute;

	// ============================================================
	// Compile GPU Kernels (once, reused across all tests)
	// ============================================================

	auto k_init_parent = device.compile<1>(
		[](BufferVar<uint> parent, const UInt n)
		{
			const UInt i = dispatch_id().x;
			$if(i < n)
			{
				parent->write(i, i);
			};
		});

	auto k_hook_edges = device.compile<1>(
		[](BufferVar<uint> row_ptr, BufferVar<uint> col_idx,
			BufferVar<uint> parent, BufferVar<uint> changed, const UInt n)
		{
			const UInt u = dispatch_id().x;
			$if(u < n)
			{
				UInt ru = u;
				$while(parent->read(ru) != ru)
				{
					ru = parent->read(ru);
				};

				UInt e_begin = row_ptr->read(u);
				UInt e_end = row_ptr->read(u + 1u);
				UInt e = e_begin;
				$while(e < e_end)
				{
					UInt v = col_idx->read(e);
					UInt rv = v;
					$while(parent->read(rv) != rv)
					{
						rv = parent->read(rv);
					};

					$if(ru != rv)
					{
						UInt hi = max(ru, rv);
						UInt lo = min(ru, rv);
						UInt old_val = parent->atomic(hi).fetch_min(lo);
						$if(old_val > lo)
						{
							changed->atomic(0u).exchange(1u);
							ru = u;
							$while(parent->read(ru) != ru)
							{
								ru = parent->read(ru);
							};
						};
					};
					e += 1u;
				};
			};
		});

	auto k_compress = device.compile<1>(
		[](BufferVar<uint> parent, const UInt n)
		{
			const UInt i = dispatch_id().x;
			$if(i < n)
			{
				UInt p = parent->read(i);
				$while(p != parent->read(p))
				{
					p = parent->read(p);
				};
				UInt root = p;

				p = i;
				$while(parent->read(p) != p)
				{
					UInt next_p = parent->read(p);
					parent->write(p, root);
					p = next_p;
				};
			};
		});

	auto k_init_parity = device.compile<1>(
		[](BufferVar<uint> comp_root, BufferVar<uint> known,
			BufferVar<uint> parity, const UInt n)
		{
			const UInt i = dispatch_id().x;
			$if(i < n)
			{
				parity->write(i, 0u);
				$if(comp_root->read(i) == i)
				{
					known->write(i, 1u);
				}
				$else
				{
					known->write(i, 0u);
				};
			};
		});

	auto k_relax_parity = device.compile<1>(
		[](BufferVar<uint> row_ptr, BufferVar<uint> col_idx,
			BufferVar<uint> flip_w, BufferVar<uint> comp_root,
			BufferVar<uint> known, BufferVar<uint> parity,
			BufferVar<uint> changed, const UInt n)
		{
			const UInt u = dispatch_id().x;
			$if(u < n)
			{
				$if(known->read(u) != 0u)
				{
					UInt pu = parity->read(u);
					UInt cu = comp_root->read(u);
					UInt e_begin = row_ptr->read(u);
					UInt e_end = row_ptr->read(u + 1u);
					UInt e = e_begin;
					$while(e < e_end)
					{
						UInt v = col_idx->read(e);
						$if(comp_root->read(v) == cu)
						{
							UInt want = (pu + flip_w->read(e)) & 1u;
							UInt old_known = known->atomic(v).compare_exchange(0u, 1u);
							$if(old_known == 0u)
							{
								parity->write(v, want);
								changed->atomic(0u).exchange(1u);
							};
						};
						e += 1u;
					};
				};
			};
		});

	auto k_check_consistency = device.compile<1>(
		[](BufferVar<uint> row_ptr, BufferVar<uint> col_idx,
			BufferVar<uint> flip_w, BufferVar<uint> comp_root,
			BufferVar<uint> known, BufferVar<uint> parity,
			BufferVar<uint> inconsistent, const UInt n)
		{
			const UInt u = dispatch_id().x;
			$if(u < n)
			{
				UInt e_begin = row_ptr->read(u);
				UInt e_end = row_ptr->read(u + 1u);
				UInt e = e_begin;
				$while(e < e_end)
				{
					UInt v = col_idx->read(e);
					$if(comp_root->read(u) == comp_root->read(v))
					{
						UInt ku = known->read(u);
						UInt kv = known->read(v);
						$if((ku & kv) == 0u)
						{
							inconsistent->atomic(0u).exchange(1u);
						}
						$else
						{
							UInt lhs = (parity->read(u) + parity->read(v)) & 1u;
							$if(lhs != flip_w->read(e))
							{
								inconsistent->atomic(0u).exchange(1u);
							};
						};
					};
					e += 1u;
				};
			};
		});

	// ============================================================
	// GPU Pipeline Runner
	// ============================================================

	struct GPUResult
	{
		std::vector<uint>			   comp_root;
		std::vector<uint>			   parity;
		std::vector<std::vector<uint>> contours;
		bool						   inconsistent = false;
		double						   elapsed_ms = 0.0;
		int							   cc_iters = 0;
		int							   parity_iters = 0;
	};

	auto run_gpu = [&](const HostCSRGraph& g) -> GPUResult
	{
		const uint N = g.num_nodes;
		const uint M = static_cast<uint>(g.col_idx.size());
		GPUResult  out;
		out.comp_root.resize(N);
		out.parity.resize(N);

		Buffer<uint> d_row_ptr = device.create_buffer<uint>(N + 1);
		Buffer<uint> d_col_idx = device.create_buffer<uint>(std::max(M, 1u));
		Buffer<uint> d_flip_w = device.create_buffer<uint>(std::max(M, 1u));
		Buffer<uint> d_parent = device.create_buffer<uint>(N);
		Buffer<uint> d_known = device.create_buffer<uint>(N);
		Buffer<uint> d_parity = device.create_buffer<uint>(N);
		Buffer<uint> d_flag = device.create_buffer<uint>(1);

		stream << d_row_ptr.copy_from(g.row_ptr.data())
			   << d_col_idx.copy_from(g.col_idx.data())
			   << d_flip_w.copy_from(g.flip_w.data())
			   << synchronize();

		luisa::Clock	  timer;
		std::vector<uint> h_flag(1);

		timer.tic();

		// Phase A: Connected Components
		stream << k_init_parent(d_parent.view(), N).dispatch(N);
		for (int iter = 0; iter < 128; ++iter)
		{
			h_flag[0] = 0u;
			stream << d_flag.copy_from(h_flag.data());
			stream << k_hook_edges(d_row_ptr.view(), d_col_idx.view(),
				d_parent.view(), d_flag.view(), N)
						  .dispatch(N);
			stream << k_compress(d_parent.view(), N).dispatch(N);
			stream << d_flag.copy_to(h_flag.data()) << synchronize();
			out.cc_iters = iter + 1;
			if (!h_flag[0])
				break;
		}
		stream << k_compress(d_parent.view(), N).dispatch(N) << synchronize();

		// Phase B: Parity Propagation
		stream << k_init_parity(d_parent.view(), d_known.view(), d_parity.view(), N).dispatch(N);
		const uint max_parity_iters = std::min(N, 10000u);
		for (uint iter = 0; iter < max_parity_iters; ++iter)
		{
			h_flag[0] = 0u;
			stream << d_flag.copy_from(h_flag.data());
			stream << k_relax_parity(d_row_ptr.view(), d_col_idx.view(), d_flip_w.view(),
				d_parent.view(), d_known.view(), d_parity.view(),
				d_flag.view(), N)
						  .dispatch(N);
			stream << d_flag.copy_to(h_flag.data()) << synchronize();
			out.parity_iters = iter + 1;
			if (!h_flag[0])
				break;
		}

		// Phase C: Consistency Check
		h_flag[0] = 0u;
		stream << d_flag.copy_from(h_flag.data());
		stream << k_check_consistency(d_row_ptr.view(), d_col_idx.view(), d_flip_w.view(),
			d_parent.view(), d_known.view(), d_parity.view(),
			d_flag.view(), N)
					  .dispatch(N);
		stream << d_flag.copy_to(h_flag.data()) << synchronize();
		out.inconsistent = (h_flag[0] != 0u);

		out.elapsed_ms = timer.toc();

		// Download
		stream << d_parent.copy_to(out.comp_root.data())
			   << d_parity.copy_to(out.parity.data())
			   << synchronize();

		// Build contours on CPU
		std::unordered_map<uint, std::vector<uint>> groups;
		for (uint i = 0; i < N; ++i)
			groups[out.comp_root[i]].push_back(i);
		for (auto& [root, members] : groups)
		{
			std::sort(members.begin(), members.end());
			out.contours.push_back(members);
		}
		std::sort(out.contours.begin(), out.contours.end());

		return out;
	};

	// ============================================================
	// Test Runner
	// ============================================================

	int total_tests = 0;
	int passed_tests = 0;

	auto run_test = [&](const char* name, const HostCSRGraph& g, uint num_runs = 3)
	{
		total_tests++;
		const uint N = g.num_nodes;
		const uint E = static_cast<uint>(g.col_idx.size()) / 2;

		// CPU reference + timing
		luisa::Clock cpu_timer;
		cpu_timer.tic();
		CPUResult cpu = cpu_reference(g);
		double	  cpu_ms = cpu_timer.toc();

		// GPU warmup
		run_gpu(g);

		// GPU timed runs
		double	  total_gpu_ms = 0.0;
		GPUResult gpu;
		for (uint r = 0; r < num_runs; ++r)
		{
			gpu = run_gpu(g);
			total_gpu_ms += gpu.elapsed_ms;
		}
		double avg_gpu_ms = total_gpu_ms / num_runs;

		// Verify
		bool passed = verify_results(g, cpu, gpu.contours, gpu.parity,
			gpu.comp_root, gpu.inconsistent, N);
		if (passed)
			passed_tests++;

		LUISA_INFO("[{}] N={} E={} contours={} inconsistent={} | "
				   "CPU: {:.4f}ms  GPU(avg): {:.4f}ms (cc={} parity={}) | {}",
			name, N, E, gpu.contours.size(), gpu.inconsistent,
			cpu_ms, avg_gpu_ms, gpu.cc_iters, gpu.parity_iters,
			passed ? "PASS" : "FAIL");
	};

	// ============================================================
	// Test Cases
	// ============================================================

	LUISA_INFO("========================================");
	LUISA_INFO("Running tests...");
	LUISA_INFO("========================================");

	std::mt19937 rng(42);

	// Test 1: Small fixed graph (original correctness test)
	{
		const uint								  N_small = 14;
		std::vector<std::tuple<uint, uint, uint>> edges = {
			{ 0, 1, 1 }, { 1, 2, 0 },					// A: chain
			{ 3, 4, 1 }, { 4, 5, 1 }, { 5, 3, 0 },		// B: consistent cycle
			{ 7, 8, 0 }, { 8, 9, 1 }, { 9, 10, 1 },		// D: chain
			{ 11, 12, 1 }, { 12, 13, 1 }, { 13, 11, 1 } // E: inconsistent triangle
		};
		run_test("small_fixed", make_undirected_csr(N_small, edges));
	}

	// Test 2: 10K chains × 10 nodes = 100K nodes, many small components
	{
		auto g = generate_chains(10000, 10, 500, rng);
		run_test("10K_chains_x10", g);
	}

	// Test 3: 1K chains × 100 nodes = 100K nodes, deeper parity propagation
	{
		auto g = generate_chains(1000, 100, 200, rng);
		run_test("1K_chains_x100", g);
	}

	// Test 4: Grid 100×100 = 10K nodes
	{
		auto g = generate_grid(100, 100, rng);
		run_test("grid_100x100", g);
	}

	// Test 5: Grid 200×250 = 50K nodes, single large component
	{
		auto g = generate_grid(200, 250, rng);
		run_test("grid_200x250", g);
	}

	// Test 6: Random sparse, 50K nodes, ~4 edges/node, 100 components
	{
		auto g = generate_random_sparse(50000, 4, 100, rng);
		run_test("random_50K_d4_c100", g);
	}

	// Test 7: Random sparse, 100K nodes, ~6 edges/node, 500 components
	{
		auto g = generate_random_sparse(100000, 6, 500, rng);
		run_test("random_100K_d6_c500", g);
	}

	// Test 8: 50K singletons + 25K pairs = 100K nodes, many trivial components
	{
		const uint								  N_tiny = 100000;
		std::vector<std::tuple<uint, uint, uint>> edges;
		std::uniform_int_distribution<uint>		  w_dist(0, 1);
		for (uint i = 50000; i < N_tiny; i += 2)
			edges.emplace_back(i, i + 1, w_dist(rng));
		run_test("singletons+pairs_100K", make_undirected_csr(N_tiny, edges));
	}

	// ============================================================
	// Summary
	// ============================================================

	LUISA_INFO("========================================");
	LUISA_INFO("Results: {}/{} tests passed.", passed_tests, total_tests);

	if (passed_tests != total_tests)
	{
		LUISA_ERROR("Some tests FAILED.");
		return EXIT_FAILURE;
	}

	LUISA_INFO("All tests PASSED.");
	return EXIT_SUCCESS;
}
