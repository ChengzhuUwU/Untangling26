// test_buffer_upload_stress.cpp
//
// Minimal, application-independent stress test for luisaCompute's CUDA buffer
// upload path. Goal: reproduce the intermittent sa_faces corruption documented
// in ResearchLog/Acceleration/ef_upload_diagnosis_d1_d6_20260722.md by varying
// the three factors that the app-layer init chain exposes to the upload pool:
//   (1) number of buffers uploaded,
//   (2) per-buffer byte size (bandwidth per upload),
//   (3) total bytes uploaded (upload-pool pressure -> staging fallback).
//
// It deliberately mirrors the app's cross-dispatch upload pattern: many
// SEPARATE `stream << ...;` statements (each a distinct dispatch) with NO
// intermediate synchronize, exactly like init_mesh_data.cpp's upload chain.
// This is the "cross-dispatch live-staging reuse" window the reviewer flagged.
//
// Usage:
//   test_buffer_upload_stress [backend=cuda] [reps=20] [trials=50]
//     reps   = independent process-style repetitions of the whole sweep
//     trials = R (per-config repeats) — corruption is probabilistic
//
// Build: cmake -DLCS_ENABLE_TEST=ON ... ; cmake --build build --target test_buffer_upload_stress

#include <cstdlib>
#include <cstring>
#include <vector>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace
{

	// Fill a host buffer with a deterministic, position-dependent pattern so
	// corruption is detectable and we can tell WHICH buffer clobbered which.
	// Each element encodes (buffer_id, element_index) so a wrong value fingers
	// the overwriter.
	template <typename T>
	void fill_pattern(std::vector<T>& v, uint buffer_id)
	{
		for (uint i = 0; i < v.size(); i++)
		{
			if constexpr (std::is_same_v<T, uint3>)
			{
				v[i] = make_uint3(buffer_id, i, buffer_id ^ i);
			}
			else if constexpr (std::is_same_v<T, uint2>)
			{
				v[i] = make_uint2(buffer_id, i);
			}
			else if constexpr (std::is_same_v<T, uint>)
			{
				v[i] = (buffer_id << 16u) | (i & 0xffffu);
			}
			else if constexpr (std::is_same_v<T, float>)
			{
				v[i] = static_cast<float>(buffer_id) * 1.5f + static_cast<float>(i) * 0.001f;
			}
			else
			{
				v[i] = static_cast<T>(buffer_id);
			}
		}
	}

	template <typename T>
	bool verify_pattern(const std::vector<T>& v, uint buffer_id, uint& first_bad_idx, T& first_bad_val)
	{
		for (uint i = 0; i < v.size(); i++)
		{
			T expected;
			if constexpr (std::is_same_v<T, uint3>)
				expected = make_uint3(buffer_id, i, buffer_id ^ i);
			else if constexpr (std::is_same_v<T, uint2>)
				expected = make_uint2(buffer_id, i);
			else if constexpr (std::is_same_v<T, uint>)
				expected = (buffer_id << 16u) | (i & 0xffffu);
			else if constexpr (std::is_same_v<T, float>)
				expected = static_cast<float>(buffer_id) * 1.5f + static_cast<float>(i) * 0.001f;
			else
				expected = static_cast<T>(buffer_id);
			// bitwise compare (floats too — pattern is exact)
			if (std::memcmp(&v[i], &expected, sizeof(T)) != 0)
			{
				first_bad_idx = i;
				first_bad_val = v[i];
				return false;
			}
		}
		return true;
	}

	// Upload N buffers of element-count `count` (type T) across SEPARATE dispatches
	// (one `stream << ...;` per buffer, no intermediate sync) — the cross-dispatch
	// pattern. Then sync once and verify every buffer. Returns # of corrupted
	// buffers; logs details on failure.
	template <typename T>
	uint run_cross_dispatch(Device& device, Stream& stream, uint N, uint count)
	{
		std::vector<Buffer<T>>		bufs;
		std::vector<std::vector<T>> host;
		bufs.reserve(N);
		host.reserve(N);
		for (uint b = 0; b < N; b++)
		{
			host.emplace_back(count);
			fill_pattern(host.back(), b);
			bufs.emplace_back(device.create_buffer<T>(count));
		}
		// Cross-dispatch: each upload is its own statement -> own dispatch.
		// No intermediate synchronize.
		for (uint b = 0; b < N; b++)
		{
			stream << bufs[b].copy_from(host[b].data());
		}
		stream << synchronize();

		// Verify each buffer.
		uint		   corrupted = 0;
		std::vector<T> tmp(count);
		for (uint b = 0; b < N; b++)
		{
			stream << bufs[b].copy_to(tmp.data()) << synchronize();
			uint bad_idx = 0;
			T	 bad_val{};
			if (!verify_pattern(tmp, b, bad_idx, bad_val))
			{
				corrupted++;
				LUISA_WARNING_WITH_LOCATION(
					"CROSS-DISPATCH CORRUPT: T={} N={} count={} buffer_id={} bad_idx={} "
					"got={} expected={}",
					sizeof(T), N, count, b, bad_idx, bad_val, host[b][bad_idx]);
			}
		}
		return corrupted;
	}

	// Same as cross-dispatch, but insert a synchronize() every `sync_every` buffers.
	// Mirrors the reviewer's targeted-sync fix (init_mesh_data.cpp:836): drain the
	// stream before the staging range can be reused by later dispatches. If the
	// root cause is cross-dispatch live-staging reuse, this should drive corrupt->0.
	template <typename T>
	uint run_cross_dispatch_synced(Device& device, Stream& stream, uint N, uint count, uint sync_every)
	{
		std::vector<Buffer<T>>		bufs;
		std::vector<std::vector<T>> host;
		bufs.reserve(N);
		host.reserve(N);
		for (uint b = 0; b < N; b++)
		{
			host.emplace_back(count);
			fill_pattern(host.back(), b);
			bufs.emplace_back(device.create_buffer<T>(count));
		}
		for (uint b = 0; b < N; b++)
		{
			stream << bufs[b].copy_from(host[b].data());
			if ((b + 1) % sync_every == 0)
				stream << synchronize();
		}
		stream << synchronize();

		uint		   corrupted = 0;
		std::vector<T> tmp(count);
		for (uint b = 0; b < N; b++)
		{
			stream << bufs[b].copy_to(tmp.data()) << synchronize();
			uint bad_idx = 0;
			T	 bad_val{};
			if (!verify_pattern(tmp, b, bad_idx, bad_val))
			{
				corrupted++;
				LUISA_WARNING_WITH_LOCATION(
					"CROSS-DISPATCH-SYNCED CORRUPT: T={} N={} count={} sync_every={} buffer_id={} "
					"bad_idx={} got={} expected={}",
					sizeof(T), N, count, sync_every, b, bad_idx, bad_val, host[b][bad_idx]);
			}
		}
		return corrupted;
	}

	// Upload the SAME N buffers within ONE `stream << a << b << ... << sync;`
	// chain (single dispatch). This is the control: single-dispatch should not
	// exhibit cross-dispatch staging reuse.
	template <typename T>
	uint run_single_chain(Device& device, Stream& stream, uint N, uint count)
	{
		std::vector<Buffer<T>>		bufs;
		std::vector<std::vector<T>> host;
		bufs.reserve(N);
		host.reserve(N);
		for (uint b = 0; b < N; b++)
		{
			host.emplace_back(count);
			fill_pattern(host.back(), b);
			bufs.emplace_back(device.create_buffer<T>(count));
		}
		// Single chain via a single stream<< expression. Build the chain by
		// repeatedly moving a Delegate. Delegate is move-only, so chain with
		// std::move on a unique_ptr-style handle is awkward; instead dispatch all
		// uploads as one CommandList via the stream directly.
		luisa::compute::CommandList cmd_list;
		for (uint b = 0; b < N; b++)
		{
			cmd_list << bufs[b].copy_from(host[b].data());
		}
		stream << std::move(cmd_list).commit() << synchronize();

		uint		   corrupted = 0;
		std::vector<T> tmp(count);
		for (uint b = 0; b < N; b++)
		{
			stream << bufs[b].copy_to(tmp.data()) << synchronize();
			uint bad_idx = 0;
			T	 bad_val{};
			if (!verify_pattern(tmp, b, bad_idx, bad_val))
			{
				corrupted++;
				LUISA_WARNING_WITH_LOCATION(
					"SINGLE-CHAIN CORRUPT: T={} N={} count={} buffer_id={} bad_idx={} "
					"got={} expected={}",
					sizeof(T), N, count, b, bad_idx, bad_val, host[b][bad_idx]);
			}
		}
		return corrupted;
	}

} // namespace

int main(int argc, char** argv)
{
	log_level_info();
	const std::string backend = (argc >= 2) ? argv[1] :
#if defined(__APPLE__)
											"metal";
#else
											"cuda";
#endif
	const uint reps = (argc >= 3) ? static_cast<uint>(std::atoi(argv[2])) : 20u;
	const uint trials = (argc >= 4) ? static_cast<uint>(std::atoi(argv[3])) : 50u;

	Context context{ std::string{ argv[0] } };
	auto	names = context.backend_device_names(backend);
	if (names.empty())
	{
		LUISA_ERROR("No {} device.", backend);
		return 1;
	}
	Device device = context.create_device(backend, nullptr, true);
	Stream stream = device.create_stream(StreamTag::COMPUTE);

	LUISA_INFO("buffer-upload-stress: backend={} reps={} trials={}", backend, reps, trials);

	// Configurations to sweep. Each: {N buffers, count elements, type size label}.
	// Mirrors the app's init chain shape: ~40 buffers, sizes from 16B to ~200KB,
	// uint3 (16B stride) for topology like sa_faces.
	struct Config
	{
		uint		N;
		uint		count;
		const char* tag;
	};
	const Config configs[] = {
		// (A) mimic app init: many small+medium buffers, uint3 topology
		{ 40, 2880, "app-like uint3 x40 count=2880" }, // ~40 x 45KB = 1.8MB
		{ 40, 1440, "app-like uint3 x40 count=1440" }, // ~40 x 22KB = 0.9MB
		// (B) upload-pool pressure: total bytes near/over the 64MiB pool
		{ 60, 8640, "pool-pressure uint3 x60 count=8640" }, // ~60 x 135KB = 8.1MB (x100 trials = lots of churn)
		// (C) many tiny buffers (CSR-like, the staging-reuse suspects)
		{ 200, 4, "many-tiny uint x200 count=4" },
		{ 500, 4, "many-tiny uint x500 count=4" },
		// (D) single big buffer (control: no cross-dispatch reuse possible)
		{ 1, 200000, "single-big uint3 count=200000" },
	};

	uint total_corrupt = 0;
	uint total_runs = 0;
	uint total_corrupt_synced = 0;

	for (uint rep = 0; rep < reps; rep++)
	{
		for (const auto& c : configs)
		{
			for (uint t = 0; t < trials; t++)
			{
				// cross-dispatch (the suspected-bad pattern)
				uint cor = run_cross_dispatch<uint3>(device, stream, c.N, c.count);
				total_corrupt += cor;
				total_runs += c.N;
				if (cor)
					LUISA_WARNING("rep={} cfg='{}' trial={} cross-dispatch corrupt={}", rep, c.tag, t, cor);

				// synced variant on the same config: sync every 8 buffers.
				// If root cause is cross-dispatch staging reuse, this should
				// eliminate corruption. Only run on small-N configs to bound time.
				if (c.N <= 60)
				{
					uint cor_s = run_cross_dispatch_synced<uint3>(device, stream, c.N, c.count, 8u);
					total_corrupt_synced += cor_s;
					if (cor_s)
						LUISA_WARNING("rep={} cfg='{}' trial={} SYNCED corrupt={} (!! fix failed)", rep, c.tag, t, cor_s);
				}

				// occasionally run the single-chain control for the same config
				if (t == 0)
				{
					uint cor2 = run_single_chain<uint3>(device, stream, c.N, c.count);
					if (cor2)
						LUISA_WARNING("rep={} cfg='{}' trial={} SINGLE-CHAIN corrupt={} (!! control)", rep, c.tag, t, cor2);
				}
			}
		}
		LUISA_INFO("rep {}/{} done: cross={} synced={} single(control last rep) runs={}",
			rep + 1, reps, total_corrupt, total_corrupt_synced, total_runs);
	}

	LUISA_INFO("=== DONE: cross-dispatch corrupt={} / synced corrupt={} / total_runs={} ===",
		total_corrupt, total_corrupt_synced, total_runs);
	return (total_corrupt == 0) ? (total_corrupt_synced == 0 ? 0 : 3) : 2;
}