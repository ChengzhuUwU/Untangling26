#include "CollisionDetector/intersection_resolver.h"
#include "CollisionDetector/intersection_resolver_gpu.h"
#include "CollisionDetector/intersection_resolver_helper.h"
#include "CollisionDetector/distance.hpp"
#include "CollisionDetector/accd.hpp"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <numeric>
#include "Utils/reduce_helper.h"
#include "luisa/core/basic_types.h"

namespace lcs
{
	void IntersectionResolver::compile(AsyncCompiler& compiler)
	{
		auto& mutex_data = host_untangling_data->vert_mutex;
		mutex_data.resize(host_mesh_data->num_verts, 0u);

		compiler.compile(fn_reset_uint,
			[](Var<Buffer<uint>> buffer)
			{
				buffer->write(luisa::compute::dispatch_x(), 0u);
			});

		compile_detection(compiler);
		compile_resolve(compiler);
		prp_shaders.compile(compiler, device_mesh_data);
	}

	void IntersectionResolver::download_intersection_list(Stream& stream)
	{
		auto& device_count = device_collision_data->narrow_phase_collision_count;
		auto& host_count = host_collision_data->narrow_phase_collision_count;

		auto& device_ef_list = device_collision_data->narrow_phase_list_ef;
		auto& device_ef_indices = device_collision_data->narrow_phase_list_ef_indices;
		auto& host_ef_list = host_collision_data->narrow_phase_list_ef;
		auto& host_ef_indices = host_collision_data->narrow_phase_list_ef_indices;

		const uint num_pairs = host_count[1];
		if (num_pairs != 0)
		{
			if (host_ef_list.size() != device_ef_list.size())
			{
				host_ef_list.resize(device_ef_list.size());
				host_ef_indices.resize(device_ef_list.size());
			}
			stream << device_ef_list.view(0, num_pairs).copy_to(host_ef_list.data())
				   << device_ef_indices.view(0, num_pairs).copy_to(host_ef_indices.data())
				   << luisa::compute::synchronize();
			CpuParallel::parallel_for(0, num_pairs, [&](uint pair_idx1)
				{
				auto ef_indices = host_ef_indices[pair_idx1];
				auto ef_pair = host_ef_list[pair_idx1];
				auto raw_indices = ef_pair.get_indices();  // std::array<uint,5>
				if (ef_indices.x > host_mesh_data->num_edges || ef_indices.y > host_mesh_data->num_faces
					|| std::any_of(raw_indices.begin(), raw_indices.end(),
				[&](uint idx) { return idx >= host_mesh_data->num_verts; }))
				{
					LUISA_ERROR("Invalid EF pair: pair_idx = {}, edge_id = {}, face_id = {}, edge = {}, face = {} . Num Verts/Edges/Faces = {}/{}/{}",
						pair_idx1,
						ef_indices.x,
						ef_indices.y,
						ef_pair.get_edge(),
						ef_pair.get_face(),
						host_mesh_data->num_verts,
						host_mesh_data->num_edges,
						host_mesh_data->num_faces);
				} });
		}
	}
} // namespace lcs

namespace lcs // Intersection Detection
{

	void IntersectionResolver::compile_detection(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;

		compiler.compile(
			fn_narrow_phase_ef_dcd_query,
			[](const BufferVar<uint>			 broad_phase_list_ef,
				BufferVar<uint>					 narrow_phase_collision_count,
				BufferVar<CollisionPair::EfPair> narrow_phase_list_ef,
				BufferVar<uint2>				 narrow_phase_list_ef_indices,
				const BufferVar<float3>			 sa_x,
				const BufferVar<uint2>			 sa_edges,
				const BufferVar<uint3>			 sa_faces,
				const BufferVar<float3>			 sa_rest_x,
				const BufferVar<float3>			 sa_x_step_start,
				const BufferVar<float>			 sa_rest_edge_area,
				const BufferVar<float>			 sa_rest_face_area,
				const BufferVar<VertexProperty>	 sa_x_property,
				const BufferVar<uint3>			 face_adj_edges,
				const BufferVar<float>			 d_hat,
				const BufferVar<float>			 thickness,
				const Uint						 max_pairs,
				const Float						 perturb_eps)
			{
				const Uint pair_idx = dispatch_id().x;

				const Uint eid = broad_phase_list_ef.read(2 * pair_idx + 0);
				const Uint fid = broad_phase_list_ef.read(2 * pair_idx + 1);

				const Uint2	 edge = sa_edges.read(eid);
				const Uint3	 face = sa_faces.read(fid);
				const Float3 Ps[2] = {
					sa_x.read(edge.x),
					sa_x.read(edge.y),
				};
				const Float3& e0 = Ps[0];
				const Float3& e1 = Ps[1];

				$if(edge[0] == face[0] | edge[0] == face[1] | edge[0] == face[2] | edge[1] == face[0]
					| edge[1] == face[1] | edge[1] == face[2])
				{
					$return();
				};

				auto left_property = sa_x_property.read(edge.x);
				auto right_property = sa_x_property.read(face.x);
				$if(left_property->is_fixed() & right_property->is_fixed())
				{
					$return();
				};
				$if(left_property->is_rigid_body() & right_property->is_rigid_body() & left_property->get_object_id() == right_property->get_object_id())
				{
					$return();
				};

				Float dhat_1 = d_hat.read(edge.x);
				Float dhat_2 = d_hat.read(face.x);
				$if(dhat_1 + dhat_2 == 0.0f) // both static
				{
					$return();
				};

				ArrayVar<Float3, 3> ABC = {
					sa_x.read(face.x),
					sa_x.read(face.y),
					sa_x.read(face.z),
				};

				auto hash_coord = [](const Uint& vid, const uint seed) -> Float
				{
					Uint x = (vid ^ seed) * 0x27d4eb2du;
					x = x ^ (x >> 15u);
					x = x * 0x85ebca6bu;
					x = x ^ (x >> 13u);
					const Float sgn = ite((x & 1u) != 0u, 1.0f, -1.0f);
					const Float mag = 0.35f + 0.65f * (cast<float>(x >> 1u) * (1.0f / 2147483647.0f));
					return sgn * mag;
				};
				auto perturb_pos = [&](const Float3& p, const Uint& vid) -> Float3
				{
					return p + make_float3(
						hash_coord(vid, 0x9e3779b9u),
						hash_coord(vid, 0x7f4a7c15u),
						hash_coord(vid, 0x6a09e667u)) * perturb_eps;
				};
				auto watertight_ef_query = [](const Float3& p0,
											   const Float3& p1,
											   const Uint	 va,
											   const Uint	 vb,
											   const Uint	 vc,
											   const Float3& a,
											   const Float3& b,
											   const Float3& c,
											   Float3&		 out_bary,
											   Float&		 out_t) -> Bool
				{
					// Canonical Plucker edge-volume test:
					// Adjacent triangles sharing an edge {lo, hi} evaluate bit-identical signed volume
					// with opposite signs and mutually exclusive tie-break, preventing 1-ULP cracks and double hits.
					Bool		 hit = false;
					const Float3 dir = p1 - p0;
					const Float3 n_tri = cross(b - a, c - a);
					const Float	 n_len = length(n_tri);
					const Float	 d_len = length(dir);
					const Float	 d0 = dot(p0 - a, n_tri);
					const Float	 d1 = dot(p1 - a, n_tri);
					const Float	 denom = d0 - d1;
					const Bool	 crosses_plane = ((d0 > 0.0f) & (d1 < 0.0f)) | ((d0 < 0.0f) & (d1 > 0.0f));

					$if(crosses_plane & (abs(denom) > 1e-12f * max(n_len * d_len, 1e-12f)))
					{
						auto edge_side = [&](const Uint u_id, const Uint v_id, const Float3& pu, const Float3& pv, Float& out_s, Bool& out_pos, Bool& out_neg)
						{
							const Bool	 fwd = u_id < v_id;
							const Float3 p_lo = ite(fwd, pu, pv);
							const Float3 p_hi = ite(fwd, pv, pu);
							const Float	 vol = dot(cross(p0 - p_lo, p1 - p_lo), p_hi - p_lo);
							out_s = ite(fwd, vol, -vol);
							out_pos = (out_s > 0.0f) | ((out_s == 0.0f) & fwd);
							out_neg = (out_s < 0.0f) | ((out_s == 0.0f) & (!fwd));
						};

						Float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
						Bool  pos0 = false, neg0 = false;
						Bool  pos1 = false, neg1 = false;
						Bool  pos2 = false, neg2 = false;
						edge_side(va, vb, a, b, s0, pos0, neg0); // opposite vc
						edge_side(vb, vc, b, c, s1, pos1, neg1); // opposite va
						edge_side(vc, va, c, a, s2, pos2, neg2); // opposite vb

						$if((pos0 & pos1 & pos2) | (neg0 & neg1 & neg2))
						{
							const Float t = d0 / denom;
							out_t = clamp(t, 1e-5f, 1.0f - 1e-5f);
							const Float w_sum = s0 + s1 + s2;
							$if(abs(w_sum) > 1e-20f)
							{
								const Float wa = clamp(s1 / w_sum, 0.0f, 1.0f);
								const Float wb = clamp(s2 / w_sum, 0.0f, 1.0f);
								const Float wc = clamp(1.0f - wa - wb, 0.0f, 1.0f);
								out_bary = make_float3(wa, wb, wc);
							}
							$else
							{
								out_bary = make_float3(1.0f / 3.0f);
							};
							hit = true;
						};
					};
					return hit;
				};

				Float3 face_bary = make_float3(0.0f);
				Float  out_time = 0.0f;
				Bool   intersect = false;

				$if(perturb_eps > 0.0f)
				{
					const Float3 e0_p = perturb_pos(e0, edge.x);
					const Float3 e1_p = perturb_pos(e1, edge.y);
					const Float3 a_p = perturb_pos(ABC[0], face.x);
					const Float3 b_p = perturb_pos(ABC[1], face.y);
					const Float3 c_p = perturb_pos(ABC[2], face.z);
					intersect = watertight_ef_query(e0_p, e1_p, face.x, face.y, face.z, a_p, b_p, c_p, face_bary, out_time);
				}
				$else
				{
					intersect = distance::LineIntersection(e0, e1, ABC[0], ABC[1], ABC[2], face_bary, out_time);
					$if(intersect)
					{
						// Suppress grazing EF contacts using normalized penetration, face barycentric, and edge-parameter thresholds.
						constexpr float k_pen_rel_eps = 1e-4f;
						constexpr float k_bary_eps = 1e-4f;
						constexpr float k_t_eps = 1e-4f;

						const Float3 N_un = luisa::compute::cross(ABC[1] - ABC[0], ABC[2] - ABC[0]);
						const Float	 N_len = luisa::compute::length(N_un);
						const Float	 face_char = luisa::compute::sqrt(luisa::compute::max(N_len, 1e-30f));
						const Float3 N = N_un / luisa::compute::max(N_len, 1e-30f);
						const Float	 d0 = luisa::compute::dot(e0 - ABC[0], N);
						const Float	 d1 = luisa::compute::dot(e1 - ABC[0], N);
						const Float	 pen = luisa::compute::min(luisa::compute::abs(d0), luisa::compute::abs(d1));
						$if(pen < k_pen_rel_eps * face_char)
						{
							intersect = false;
						};

						const Float bary_min = luisa::compute::min(luisa::compute::min(face_bary.x, face_bary.y), face_bary.z);
						$if(bary_min < k_bary_eps)
						{
							intersect = false;
						};

						const Float t_min = luisa::compute::min(out_time, 1.0f - out_time);
						$if(t_min < k_t_eps)
						{
							intersect = false;
						};
					};
				};

				$if(intersect)
				{
					Float2 edge_bary = { 1.0f - out_time, out_time };

					Var<CollisionPair::EfPair> ef_pair;
					ef_pair->make_ef_pair(edge, face, edge_bary, face_bary);
					ef_pair->set_gradient(make_float3(0.0f));
					ef_pair->set_stiffness(0.0f);

					const Uint pair_idx = narrow_phase_collision_count.atomic(1u).fetch_add(1u);
					$if(pair_idx < max_pairs)
					{
						narrow_phase_list_ef_indices.write(pair_idx, make_uint2(eid, fid));
						narrow_phase_list_ef.write(pair_idx, ef_pair);
					};
				};
			});

		compiler.compile(
			fn_add_ef_from_proximity_pairs,
			[](Var<CDBG>				collision_data,
				const BufferVar<float3> sa_x,
				const BufferVar<uint2>	sa_edges,
				const BufferVar<uint3>	sa_faces,
				const BufferVar<uint>	vert_adj_edges_csr,
				// const BufferVar<uint2>	edge_adj_faces,
				const BufferVar<uint>  edge_adj_faces_csr,
				const BufferVar<float> sa_per_vert_offset,
				const Uint			   max_proximity_pairs,
				const Uint			   max_pairs)
			{
				const Uint pair_idx = dispatch_id().x;
				const Uint num_proximity_pairs = luisa::compute::min(collision_data.narrow_phase_collision_count.read(0u), max_proximity_pairs);
				$if(pair_idx >= num_proximity_pairs)
				{
					$return();
				};

				auto	   pair = collision_data.narrow_phase_list.read(pair_idx);
				const Uint type = pair->get_collision_type();
				$if(type != CollisionPair::type_vf() & type != CollisionPair::type_ee())
				{
					$return();
				};

				const Uint4	 indices = pair->get_indices();
				const Float4 weight = pair->get_weight();
				Float3		 delta = weight[0] * sa_x.read(indices[0]) + weight[1] * sa_x.read(indices[1])
					+ weight[2] * sa_x.read(indices[2]) + weight[3] * sa_x.read(indices[3]);
				Float d2 = dot(delta, delta);
				// Float thickness = sa_per_vert_offset.read(indices[0]) + sa_per_vert_offset.read(indices[2]);
				Float thickness = 0.0f;
				$if(d2 >= thickness * thickness + 1e-10f)
				{
					$return();
				};

				Var<CollisionPair::EfPair> ef_pair;

				$if(type == CollisionPair::type_vf())
				{
					const Uint	 vid = indices[0];
					const Uint	 fid = collision_data.narrow_phase_list_indices.read(pair_idx).y;
					const Uint3	 face = sa_faces.read(fid);
					const Float3 face_bary = abs(pair->get_weight().yzw());

					const Uint prefix_ae = vert_adj_edges_csr.read(vid);
					const Uint suffix_ae = vert_adj_edges_csr.read(vid + 1u);
					$for(ei, suffix_ae - prefix_ae)
					{
						const Uint	ae = vert_adj_edges_csr.read(prefix_ae + ei);
						const Uint2 adj_edge = sa_edges.read(ae);

						const Uint ef_pair_idx = collision_data.narrow_phase_collision_count.atomic(1u).fetch_add(1u);
						$if(ef_pair_idx < max_pairs)
						{
							Float		 edge_bary0 = ite(adj_edge.x == vid, 1.0f, 0.0f);
							const Float2 edge_bary = make_float2(edge_bary0, 1.0f - edge_bary0);
							ef_pair->make_ef_pair(adj_edge, face, edge_bary, face_bary);
							ef_pair->set_gradient(make_float3(0.0f));
							ef_pair->set_stiffness(0.0f);
							collision_data->narrow_phase_list_ef_indices.write(ef_pair_idx, make_uint2(ae, fid));
							collision_data->narrow_phase_list_ef.write(ef_pair_idx, ef_pair);
						};
					};
				}
				$else
				{
					const Uint2	 proximity_pair_indices = collision_data.narrow_phase_list_indices.read(pair_idx);
					const Uint	 eid1 = proximity_pair_indices.x;
					const Uint	 eid2 = proximity_pair_indices.y;
					const Uint2	 edge1 = sa_edges.read(eid1);
					const Uint2	 edge_2 = sa_edges.read(eid2);
					const Float2 edge1_bary = abs(pair->get_weight().xy());
					const Float2 edge2_bary = abs(pair->get_weight().zw());

					{
						const Uint prefix_af = edge_adj_faces_csr.read(eid2);
						const Uint suffix_af = edge_adj_faces_csr.read(eid2 + 1u);
						$for(fi, suffix_af - prefix_af)
						{
							const Uint	af = edge_adj_faces_csr.read(prefix_af + fi);
							const Uint3 face = sa_faces.read(af);
							Bool		shares_left_edge_vert = any(face == edge1.x) | any(face == edge1.y);
							$if(!shares_left_edge_vert)
							{
								Float3 face_bary = make_float3(0.0f);
								Uint   matched_vert_count = 0u;
								for (uint ii = 0; ii < 3; ii++)
								{
									$if(face[ii] == edge_2.x)
									{
										face_bary[ii] = edge2_bary.x;
										matched_vert_count += 1u;
									}
									$elif(face[ii] == edge_2.y)
									{
										face_bary[ii] = edge2_bary.y;
										matched_vert_count += 1u;
									};
								}
								$if(matched_vert_count == 2u)
								{
									ef_pair->make_ef_pair(edge1, face, edge1_bary, face_bary);
									ef_pair->set_gradient(make_float3(0.0f));
									ef_pair->set_stiffness(0.0f);

									const Uint ef_pair_idx = collision_data.narrow_phase_collision_count.atomic(1u).fetch_add(1u);
									$if(ef_pair_idx < max_pairs)
									{
										collision_data->narrow_phase_list_ef_indices.write(ef_pair_idx, make_uint2(eid1, af));
										collision_data->narrow_phase_list_ef.write(ef_pair_idx, ef_pair);
									};
								};
							};
						};
					}
					{

						const Uint prefix_af = edge_adj_faces_csr.read(eid1);
						const Uint suffix_af = edge_adj_faces_csr.read(eid1 + 1u);
						$for(fi, suffix_af - prefix_af)
						{
							const Uint	af = edge_adj_faces_csr.read(prefix_af + fi);
							const Uint3 face = sa_faces.read(af);
							Bool		shares_right_edge_vert = any(face == edge_2.x) | any(face == edge_2.y);
							$if(!shares_right_edge_vert)
							{
								Float3 face_bary = make_float3(0.0f);
								Uint   matched_vert_count = 0u;
								for (uint ii = 0; ii < 3; ii++)
								{
									$if(face[ii] == edge1.x)
									{
										face_bary[ii] = edge1_bary.x;
										matched_vert_count += 1u;
									}
									$elif(face[ii] == edge1.y)
									{
										face_bary[ii] = edge1_bary.y;
										matched_vert_count += 1u;
									};
								}
								$if(matched_vert_count == 2u)
								{
									ef_pair->make_ef_pair(edge_2, face, edge2_bary, face_bary);
									ef_pair->set_gradient(make_float3(0.0f));
									ef_pair->set_stiffness(0.0f);

									const Uint ef_pair_idx = collision_data.narrow_phase_collision_count.atomic(1u).fetch_add(1u);
									$if(ef_pair_idx < max_pairs)
									{
										collision_data->narrow_phase_list_ef_indices.write(ef_pair_idx, make_uint2(eid2, af));
										collision_data->narrow_phase_list_ef.write(ef_pair_idx, ef_pair);
									};
								};
							};
						};
					}

					// Legacy EF-pair fallback construction retained for reference.
				};
			});
	}
	void IntersectionResolver::ef_dcd_intersection_query(Stream& stream,
		const Buffer<float3>&									 sa_x,
		const Buffer<uint2>&									 sa_edges,
		const Buffer<uint3>&									 sa_faces,
		const Buffer<float3>&									 sa_rest_x,
		const Buffer<float3>&									 sa_x_step_start,
		const Buffer<float>&									 sa_rest_edge_area,
		const Buffer<float>&									 sa_rest_face_area,
		const Buffer<VertexProperty>&							 sa_x_property,
		const Buffer<float>&									 d_hat,
		const Buffer<float>&									 thickness,
		const float												 kappa)
	{
		const auto& host_count = host_collision_data->broad_phase_collision_count;
		const uint	num_broadphase_vf = host_count[host_collision_data->get_vf_count_offset()];
		const uint	num_broadphase_ee = host_count[host_collision_data->get_ee_count_offset()];
		const uint	num_broadphase_ef = host_count[host_collision_data->get_ef_count_offset()];

		// Compute deterministic symbolic-perturbation epsilon for GIA mode (0 for PRP/ICM to preserve exact baseline behavior).
		float perturb_eps = 0.0f;
		if (get_scene_params().use_untangling_GIA)
		{
			if (host_mesh_data != nullptr && host_mesh_data->num_edges > 0u
				&& (gia_cached_num_edges != host_mesh_data->num_edges || gia_cached_avg_edge_len <= 0.0f))
			{
				double sum_len = 0.0;
				for (uint eid = 0u; eid < host_mesh_data->num_edges; ++eid)
				{
					const uint2 e = host_mesh_data->sa_edges[eid];
					sum_len += static_cast<double>(luisa::length(host_mesh_data->sa_rest_x[e.y] - host_mesh_data->sa_rest_x[e.x]));
				}
				gia_cached_avg_edge_len = static_cast<float>(sum_len / static_cast<double>(host_mesh_data->num_edges));
				gia_cached_num_edges = host_mesh_data->num_edges;
			}
			perturb_eps = std::clamp(2e-4f * gia_cached_avg_edge_len, 1e-5f, 1e-4f);
		}
		gia_last_perturb_eps = perturb_eps;

		// host_mesh_data->face_adj_edges
		stream << fn_reset_uint(device_collision_data->vert_is_invalid)
					  .dispatch(device_collision_data->vert_is_invalid.size());
		stream << fn_narrow_phase_ef_dcd_query(device_collision_data->broad_phase_list_ef,
			device_collision_data->narrow_phase_collision_count,
			device_collision_data->narrow_phase_list_ef,
			device_collision_data->narrow_phase_list_ef_indices,
			sa_x,
			sa_edges,
			sa_faces,
			sa_rest_x,
			sa_x_step_start,
			sa_rest_edge_area,
			sa_rest_face_area,
			sa_x_property,
			device_mesh_data->face_adj_edges,
			d_hat,
			thickness,
			get_collision_data().narrow_phase_list_ef.size(),
			perturb_eps)
					  .dispatch(num_broadphase_ef);

		if (!get_scene_params().ignore_near_zero_dist_pairs)
		{
			const uint num_proximity_upper_bound = num_broadphase_vf + num_broadphase_ee;
			const uint max_proximity_pairs = static_cast<uint>(get_collision_data().narrow_phase_list.size());
			const uint proximity_dispatch_count = std::min(num_proximity_upper_bound, max_proximity_pairs);
			if (proximity_dispatch_count != 0u)
			{
				stream << fn_add_ef_from_proximity_pairs(get_collision_data(),
					sa_x,
					sa_edges,
					sa_faces,
					device_mesh_data->sa_vert_adj_edges_csr,
					device_mesh_data->edge_adj_faces_csr,
					thickness,
					max_proximity_pairs,
					get_collision_data().narrow_phase_list_ef.size())
							  .dispatch(proximity_dispatch_count);
			}
		}
	}

} // namespace lcs

namespace lcs // Contour preprocessing
{

	// Output: loop-vertex/pair mapping plus EF pairs and their indices.
	void find_loop_vertex(const uint		  eid,
		const uint							  fid,
		const uint							  pair_idx,
		const std::vector<uint2>&			  sa_edges,
		const std::vector<uint3>&			  sa_faces,
		const std::vector<std::vector<uint>>& sa_edge_adj_edges_ext,
		const std::vector<std::vector<uint>>& sa_edge_adj_faces_ext,
		const std::vector<std::vector<uint>>& sa_face_adj_faces_ext,
		const std::vector<uint3>&			  sa_face_adj_edges,
		const std::vector<float3>&			  positions,
		std::vector<uint2>&					  ef_indices,
		std::vector<uint2>&					  loop_pairs_vertex,
		const uint							  num_pairs)
	{
		// via loop vertex
		{
			uint2		edge = sa_edges[eid];
			uint3		face = sa_faces[fid];
			const auto& edge_adj_faces = sa_edge_adj_faces_ext[eid];
			for (const uint adj_fid : edge_adj_faces)
			{
				if (adj_fid != -1u) //  && face_contains_pairs_list_ref[adj_fid].empty()
				{
					const uint3 adj_face = sa_faces[adj_fid];
					const uint	adj_vid = adj_face[0] + adj_face[1] + adj_face[2] - edge.x - edge.y;
					if (adj_vid == face[0] || adj_vid == face[1] || adj_vid == face[2]) // Loop-vertex
					{
						// LUISA_INFO("Found loop vertex {} for pair {}, edge {}, face {}, adj_face {}", adj_vid, pair_idx, edge, face, adj_face);
						const uint loop_vid = adj_vid;
						const uint loop_fid = fid;
						uint	   loop_eid;
						{
							const uint3 face_adj_eids = sa_face_adj_edges[adj_fid];
							uint2		face_adj_edges[3] = {
								sa_edges[face_adj_eids[0]],
								sa_edges[face_adj_eids[1]],
								sa_edges[face_adj_eids[2]],
							};
							if (eid == face_adj_eids[0])
							{
								if (luisa::any(face_adj_edges[1] == loop_vid))
									loop_eid = face_adj_eids[1];
								else
									loop_eid = face_adj_eids[2];
							}
							else if (eid == face_adj_eids[1])
							{
								if (luisa::any(face_adj_edges[0] == loop_vid))
									loop_eid = face_adj_eids[0];
								else
									loop_eid = face_adj_eids[2];
							}
							else if (eid == face_adj_eids[2])
							{
								if (luisa::any(face_adj_edges[0] == loop_vid))
									loop_eid = face_adj_eids[0];
								else
									loop_eid = face_adj_eids[1];
							}
							else
							{
								LUISA_ERROR("Edge {} not adjacent to face {}", eid, adj_fid);
							}
						}
						loop_pairs_vertex.push_back({ loop_vid, pair_idx });
						ef_indices.push_back(luisa::make_uint2(loop_eid, loop_fid));
					}
				}
			}
		}
	}

	std::vector<uint> compute_adj_pairs(const uint eid,
		const uint								   fid,
		const uint								   pair_idx,
		const std::vector<std::vector<uint>>&	   edge_contains_pairs_list_ref,
		const std::vector<std::vector<uint>>&	   face_contains_pairs_list_ref,
		const std::vector<std::vector<uint>>&	   sa_edge_adj_edges_ext,
		const std::vector<std::vector<uint>>&	   sa_edge_adj_faces_ext,
		const std::vector<uint2>&				   ef_indices)
	{
		std::vector<uint> adj_pairs;

		// via edge's adjacent edges
		{
			for (const uint adj_eid : sa_edge_adj_edges_ext[eid])
			{
				if (adj_eid != -1u)
				{
					const std::vector<uint>& adj_edge_contains_pairs = edge_contains_pairs_list_ref[adj_eid];
					for (const uint& adj_pair_idx : adj_edge_contains_pairs)
					{
						const uint adj_fid = ef_indices[adj_pair_idx].y;
						if (adj_fid == fid && adj_pair_idx != pair_idx)
						{
							adj_pairs.push_back(adj_pair_idx);
						}
					}
				}
			}
		}

		// via edge's adjacent faces
		{
			const auto&				 adj_faces = sa_edge_adj_faces_ext[eid];
			const std::vector<uint>& intersection_face_contains_pairs = face_contains_pairs_list_ref[fid];
			for (const uint& adj_pair_idx : intersection_face_contains_pairs)
			{
				if (adj_pair_idx != pair_idx)
				{
					const uint adj_fid = ef_indices[adj_pair_idx].y;
					for (const uint adj_face : adj_faces)
					{
						if (adj_fid == adj_face)
						{
							const uint adj_eid = ef_indices[adj_pair_idx].x;
							if (adj_eid != eid && adj_pair_idx != pair_idx)
							{
								adj_pairs.push_back(adj_pair_idx);
							}
						}
					}
				}
			}
		}

		return adj_pairs;
	};

	// DCD rejects boundary hits here, while proximity VF/EE conversion deliberately places a barycentric coordinate on the boundary.
	constexpr float k_degenerate_ef_bary_eps = 1e-4f;

	[[nodiscard]] bool is_degenerate_ef_pair(const CollisionPair::EfPair& pair)
	{
		const float2 edge_bary = pair.get_edge_bary();
		const float3 face_bary = pair.get_face_bary();
		return std::min(edge_bary.x, edge_bary.y) < k_degenerate_ef_bary_eps
			|| std::min({ face_bary.x, face_bary.y, face_bary.z }) < k_degenerate_ef_bary_eps;
	}

	// Pre-process EF pairs:
	constexpr float UE_SMALL_NUMBER = 1.e-8f;

	struct ICMLocalGradientTerms
	{
		float3 tangent = luisa::make_float3(0.0f);
		float3 normal = luisa::make_float3(0.0f);
		float3 face_normal = luisa::make_float3(0.0f);
	};

	inline ICMLocalGradientTerms compute_local_gradient_terms(const uint eid,
		const uint														 fid,
		const CollisionPair::EfPair&									 pair,
		const std::vector<std::vector<uint>>&							 sa_edge_adj_faces_ext,
		const std::vector<uint2>&										 sa_edges,
		const std::vector<uint3>&										 sa_faces,
		const std::vector<float3>&										 sa_x)
	{
		const uint2	 edge = sa_edges[eid];
		const uint3	 face = sa_faces[fid];
		const float3 edge_positions[2] = {
			sa_x[edge.x],
			sa_x[edge.y],
		};
		const float3 face_positions[3] = {
			sa_x[face.x],
			sa_x[face.y],
			sa_x[face.z],
		};

		const float3 f0 = face_positions[0];
		const float3 f1 = face_positions[1];
		const float3 f2 = face_positions[2];
		float3		 face_normal = luisa::cross(f1 - f0, f2 - f0);
		const float	 face_normal_len = luisa::sqrt(luisa::dot(face_normal, face_normal));
		if (face_normal_len > UE_SMALL_NUMBER)
		{
			face_normal = face_normal * (1.0f / face_normal_len);
		}

		const float3		  EdgePos0 = edge_positions[0];
		const float3		  EdgePos1 = edge_positions[1];
		const float3		  EdgeDir = EdgePos1 - EdgePos0;
		const float			  EdgeDirDotN = luisa::dot(EdgeDir, face_normal);
		ICMLocalGradientTerms terms;
		terms.face_normal = face_normal;
		if (luisa::abs(EdgeDirDotN) >= UE_SMALL_NUMBER)
		{
			const float3 NOverEDotN = face_normal * (1.0f / EdgeDirDotN);

			const auto& edge_to_faces = sa_edge_adj_faces_ext[eid];
			for (const uint EdgeFace_i : edge_to_faces)
			{
				if (EdgeFace_i == -1u)
					continue;

				const uint3	 Bi = sa_faces[EdgeFace_i];
				const float3 P0 = sa_x[Bi.x];
				const float3 P1 = sa_x[Bi.y];
				const float3 P2 = sa_x[Bi.z];

				float3		Ni = luisa::cross(P1 - P0, P2 - P0);
				const float Ni_len = luisa::sqrt(luisa::dot(Ni, Ni));
				if (Ni_len <= UE_SMALL_NUMBER)
					continue;
				Ni = Ni * (1.0f / Ni_len);

				float3		Ri = luisa::cross(face_normal, Ni);
				const float R_len = luisa::sqrt(luisa::dot(Ri, Ri));
				if (R_len <= UE_SMALL_NUMBER)
					continue;
				Ri = Ri * (1.0f / R_len);

				const uint edge_v0 = edge.x;
				const uint edge_v1 = edge.y;
				int		   EdgePointLocalIndex0 = edge_v0 == Bi.x ? 0 : edge_v0 == Bi.y ? 1
																						: 2;
				const uint next_idx = Bi[(EdgePointLocalIndex0 + 1) % 3];
				const bool bEdgePointsFlipped = (edge_v1 != next_idx);

				const float TestFlipRValue = (luisa::dot(face_normal, Ni) * luisa::dot(Ni, EdgeDir) - luisa::dot(face_normal, EdgeDir)) * (bEdgePointsFlipped ? -1.0f : 1.0f);
				if (TestFlipRValue < 0.f)
				{
					Ri *= -1.0f;
				}

				terms.tangent += Ri;
				terms.normal += -2.0f * luisa::dot(EdgeDir, Ri) * NOverEDotN;
			}
		}
		return terms;
	}

	inline float3 compute_local_gradient(const uint eid,
		const uint									fid,
		const CollisionPair::EfPair&				pair,
		const std::vector<std::vector<uint>>&		sa_edge_adj_faces_ext,
		const std::vector<uint2>&					sa_edges,
		const std::vector<uint3>&					sa_faces,
		const std::vector<float3>&					sa_x)
	{
		const ICMLocalGradientTerms terms = compute_local_gradient_terms(eid,
			fid,
			pair,
			sa_edge_adj_faces_ext,
			sa_edges,
			sa_faces,
			sa_x);
		return terms.tangent + terms.normal;
	}

	// Pre-process EF pairs: sort, filter degenerates, detect loops, build adjacency, and compute positions/area.
	void IntersectionResolver::host_preprocess_ef_pairs(Device& device, Stream& stream)
	{
		auto& host_count = host_collision_data->narrow_phase_collision_count;
		uint& num_pairs = host_count[1];

		std::vector<uint2>				   ef_indices(host_collision_data->narrow_phase_list_ef_indices.begin(),
			host_collision_data->narrow_phase_list_ef_indices.begin() + num_pairs);
		std::vector<CollisionPair::EfPair> ef_list(host_collision_data->narrow_phase_list_ef.begin(),
			host_collision_data->narrow_phase_list_ef.begin() + num_pairs);

		// Always keep a deterministic edge/face ordering for contour construction (including GIA).
		{
			std::vector<uint2>				   tmp_ef_indices(ef_indices.begin(), ef_indices.end());
			std::vector<CollisionPair::EfPair> tmp_ef_list(ef_list.begin(), ef_list.end());

			std::vector<uint> sorted_get_original(num_pairs);
			std::iota(sorted_get_original.begin(), sorted_get_original.end(), 0);

			CpuParallel::parallel_sort(sorted_get_original.begin(),
				sorted_get_original.end(),
				[&tmp_ef_indices, &tmp_ef_list](const uint ii, const uint jj)
				{
					const auto left = tmp_ef_indices[ii];
					const auto right = tmp_ef_indices[jj];
					if (left[0] != right[0])
						return left[0] < right[0];
					if (left[1] != right[1])
						return left[1] < right[1];
					// Duplicate (edge, face) keys occur (e.g. two intersection
					// segments of one edge-face pair); break ties on the payload so
					// the unstable parallel sort still yields a canonical order.
					static_assert(std::is_trivially_copyable_v<CollisionPair::EfPair>);
					return std::memcmp(&tmp_ef_list[ii], &tmp_ef_list[jj], sizeof(CollisionPair::EfPair)) < 0;
				});

			CpuParallel::parallel_for(0,
				num_pairs,
				[&](const uint sorted_idx)
				{
					const uint pair_idx = sorted_get_original[sorted_idx];
					ef_indices[sorted_idx] = tmp_ef_indices[pair_idx];
					ef_list[sorted_idx] = tmp_ef_list[pair_idx];
				});

			// Temporary [DET] determinism trace (bisecting synthetic run-to-run
			// divergence); gated on prp_debug and slated for removal after diagnosis.
			if (get_scene_params().prp_debug)
			{
				auto hash_bytes = [](uint64_t h, const void* data, size_t n)
				{
					const auto* p = static_cast<const unsigned char*>(data);
					for (size_t i = 0; i < n; i++)
					{
						h ^= p[i];
						h *= 1099511628211ull;
					}
					return h;
				};
				uint64_t h = 1469598103934665603ull;
				h = hash_bytes(h, &num_pairs, 4);
				h = hash_bytes(h, ef_indices.data(), ef_indices.size() * sizeof(uint2));
				h = hash_bytes(h, ef_list.data(), ef_list.size() * sizeof(CollisionPair::EfPair));
				LUISA_INFO("[DET] P0_sorted_ef {:016x}", h);

				// P0a: order-sensitive hash of the sort keys only.
				uint64_t ha = 1469598103934665603ull;
				ha = hash_bytes(ha, &num_pairs, 4);
				ha = hash_bytes(ha, ef_indices.data(), ef_indices.size() * sizeof(uint2));
				// Count duplicate sort keys (same edge,face) to detect tie instability.
				uint64_t duplicate_key_count = 0u;
				for (uint i = 1u; i < num_pairs; i++)
					if (ef_indices[i][0] == ef_indices[i - 1u][0]
						&& ef_indices[i][1] == ef_indices[i - 1u][1])
						duplicate_key_count++;
				ha = hash_bytes(ha, &duplicate_key_count, 8);
				LUISA_INFO("[DET] P0a_sorted_indices {:016x} (duplicate_keys={})", ha, duplicate_key_count);

				// P0b: order-INSENSITIVE canonical multiset hash of key+payload bytes.
				struct KeyedPair
				{
					uint2								   indices;
					std::array<unsigned char, sizeof(CollisionPair::EfPair)> payload;
				};
				std::vector<KeyedPair> keyed(num_pairs);
				for (uint i = 0u; i < num_pairs; i++)
				{
					keyed[i].indices = ef_indices[i];
					std::memcpy(keyed[i].payload.data(), &ef_list[i], sizeof(CollisionPair::EfPair));
				}
				std::sort(keyed.begin(), keyed.end(),
					[](const KeyedPair& l, const KeyedPair& r)
					{
						if (l.indices[0] != r.indices[0])
							return l.indices[0] < r.indices[0];
						if (l.indices[1] != r.indices[1])
							return l.indices[1] < r.indices[1];
						return std::memcmp(l.payload.data(), r.payload.data(), l.payload.size()) < 0;
					});
				uint64_t hb = 1469598103934665603ull;
				hb = hash_bytes(hb, &num_pairs, 4);
				for (const auto& k : keyed)
				{
					hb = hash_bytes(hb, &k.indices, sizeof(uint2));
					hb = hash_bytes(hb, k.payload.data(), k.payload.size());
				}
				LUISA_INFO("[DET] P0b_canonical_multiset {:016x}", hb);
			}
		}

		auto build_ef_pair_adjacency = [&](const std::vector<uint2>& pair_indices)
		{
			const uint					   pair_count = static_cast<uint>(pair_indices.size());
			std::vector<std::vector<uint>> edge_contains_pairs_list(host_mesh_data->num_edges);
			std::vector<std::vector<uint>> face_contains_pairs_list(host_mesh_data->num_faces);
			std::vector<std::vector<uint>> adjacency(pair_count);

			for (uint pair_idx = 0; pair_idx < pair_count; pair_idx++)
			{
				const uint2 pair = pair_indices[pair_idx];
				edge_contains_pairs_list[pair.x].push_back(pair_idx);
			}
			CpuParallel::parallel_for(0,
				host_mesh_data->num_faces,
				[&](const uint fid)
				{
					const uint3		  face_adj_edges = host_mesh_data->face_adj_edges[fid];
					std::vector<uint> contains_pairs;
					for (uint ii = 0; ii < 3; ii++)
					{
						const uint	eid = face_adj_edges[ii];
						const auto& edge_pairs = edge_contains_pairs_list[eid];
						contains_pairs.insert(contains_pairs.end(),
							edge_pairs.begin(),
							edge_pairs.end());
					}
					face_contains_pairs_list[fid] = std::move(contains_pairs);
				});

			CpuParallel::parallel_for(0,
				pair_count,
				[&](const uint pair_idx)
				{
					adjacency[pair_idx] = compute_adj_pairs(
						pair_indices[pair_idx].x,
						pair_indices[pair_idx].y,
						pair_idx,
						edge_contains_pairs_list,
						face_contains_pairs_list,
						host_mesh_data->edge_adj_edges_ext,
						host_mesh_data->edge_adj_faces_ext,
						pair_indices);
				});
			return adjacency;
		};

		std::vector<std::vector<uint>> ef_pair_adj_pairs_ext = build_ef_pair_adjacency(ef_indices);

		if (!get_scene_params().ignore_near_zero_dist_pairs)
		{
			std::vector<luisa::ubyte> pair_is_degenerate(num_pairs, 0u);
			uint					  degenerate_candidate_count = 0u;
			for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
			{
				pair_is_degenerate[pair_idx] = is_degenerate_ef_pair(ef_list[pair_idx]) ? 1u : 0u;
				degenerate_candidate_count += pair_is_degenerate[pair_idx] != 0u ? 1u : 0u;
			}

			if (degenerate_candidate_count != 0u)
			{
				std::vector<uint2>				   filtered_ef_indices;
				std::vector<CollisionPair::EfPair> filtered_ef_list;
				filtered_ef_indices.reserve(num_pairs);
				filtered_ef_list.reserve(num_pairs);
				uint kept_degenerate_count = 0u;

				for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
				{
					bool keep_pair = pair_is_degenerate[pair_idx] == 0u;
					if (!keep_pair)
					{
						keep_pair = std::any_of(ef_pair_adj_pairs_ext[pair_idx].begin(),
							ef_pair_adj_pairs_ext[pair_idx].end(),
							[&](const uint adj_pair_idx)
							{ return pair_is_degenerate[adj_pair_idx] == 0u; });
						kept_degenerate_count += keep_pair ? 1u : 0u;
					}

					if (keep_pair)
					{
						filtered_ef_indices.push_back(ef_indices[pair_idx]);
						filtered_ef_list.push_back(ef_list[pair_idx]);
					}
				}

				if (filtered_ef_indices.empty() && num_pairs > 0u)
				{
					LUISA_INFO("All {} EF pairs are boundary/degenerate (pure boundary intersection complex); retaining all pairs for contour solver.", num_pairs);
				}
				else
				{
					const uint removed_degenerate_count = degenerate_candidate_count - kept_degenerate_count;
					LUISA_INFO("Degenerate EF candidates from proximity search: {} total, {} adjacent to non-degenerate EF pairs, {} isolated and removed",
						degenerate_candidate_count,
						kept_degenerate_count,
						removed_degenerate_count);

					if (removed_degenerate_count != 0u)
					{
						ef_indices = std::move(filtered_ef_indices);
						ef_list = std::move(filtered_ef_list);
						num_pairs = static_cast<uint>(ef_indices.size());
						ef_pair_adj_pairs_ext = build_ef_pair_adjacency(ef_indices);
					}
				}
			}
		}

		if (num_pairs == 0u)
		{
			host_collision_data->narrow_phase_list_ef_indices.clear();
			host_collision_data->narrow_phase_list_ef.clear();
			host_untangling_data->loop_pairs_vertex.clear();
			host_untangling_data->ef_pair_adj_pairs.clear();
			host_untangling_data->ef_pair_adj_pairs_ext.clear();
			host_untangling_data->ef_pair_pos_2D.clear();
			host_untangling_data->ef_pair_pos_3D.clear();
			host_untangling_data->ef_pair_length.clear();
			stream << device_collision_data->narrow_phase_collision_count.view(1, 1).copy_from(&num_pairs)
				   << luisa::compute::synchronize();
			return;
		}

		// Find loop vertices and augment adjacency with synthetic loop pairs.
		std::vector<uint2> ef_pair_adj_pairs(num_pairs, luisa::make_uint2(-1u, -1u));
		std::vector<uint2> loop_pairs_vertex;
		{

			// Process loop vertex and update adjacent pairs with loop vertex pairs
			{
				loop_pairs_vertex.clear();
		std::vector<uint2> loop_pair_ef_indices; // indexed by loop_index (parallel to loop_pairs_vertex)
		for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
		{
			find_loop_vertex(ef_indices[pair_idx].x,
				ef_indices[pair_idx].y,
				pair_idx,
				host_mesh_data->sa_edges,
				host_mesh_data->sa_faces,
				host_mesh_data->edge_adj_edges_ext,
				host_mesh_data->edge_adj_faces_ext,
				host_mesh_data->face_adj_faces_ext,
				host_mesh_data->face_adj_edges,
				host_sim_data->sa_x,
				loop_pair_ef_indices,
				loop_pairs_vertex,
				num_pairs);
		}

		const uint num_pairs_orig = num_pairs;

		if (get_scene_params().use_untangling_GIA)
		{
			std::map<uint, std::vector<uint>> loop_vertex_contains_pairs;
			for (uint loop_index = 0; loop_index < loop_pairs_vertex.size(); loop_index++)
			{
				const uint loop_vertex = loop_pairs_vertex[loop_index].x;
				loop_vertex_contains_pairs[loop_vertex].push_back(loop_index);
			}
			std::vector<uint2> final_loop_pairs_vertex;
			for (const auto& [loop_vertex, loop_indices] : loop_vertex_contains_pairs)
			{
				std::vector<uint> adj_normal_pairs;
				adj_normal_pairs.reserve(loop_indices.size());
				for (const uint loop_index : loop_indices)
					adj_normal_pairs.push_back(loop_pairs_vertex[loop_index].y);

				const uint first_loop_index = loop_indices[0];
				const uint loop_eid = loop_pair_ef_indices[first_loop_index].x;
				const uint loop_fid = loop_pair_ef_indices[first_loop_index].y;
				const uint new_loop_pair_idx = static_cast<uint>(ef_list.size());

				CollisionPair::EfPair pair;
				{
					const uint2	 loop_edge = luisa::make_uint2(loop_vertex, loop_vertex);
					const uint3	 loop_face = luisa::make_uint3(loop_vertex, loop_vertex, loop_vertex);
					const float2 edge_bary = luisa::make_float2(1.0f, 0.0f);
					const float3 face_bary = luisa::make_float3(1.0f, 0.0f, 0.0f);
					pair.make_ef_pair(loop_edge, loop_face, edge_bary, face_bary);
					pair.set_gradient(luisa::make_float3(0.0f));
					pair.set_stiffness(0.0f);
				}
				ef_list.push_back(pair);
				ef_indices.push_back(luisa::make_uint2(loop_eid, loop_fid));
				ef_pair_adj_pairs_ext.push_back(adj_normal_pairs);

				for (const uint normal_pair_idx : adj_normal_pairs)
					ef_pair_adj_pairs_ext[normal_pair_idx].push_back(new_loop_pair_idx);

				final_loop_pairs_vertex.push_back(luisa::make_uint2(loop_vertex, loop_pairs_vertex[first_loop_index].y));
			}
			loop_pairs_vertex = std::move(final_loop_pairs_vertex);
		}
		else
		{
			for (uint loop_index = 0; loop_index < loop_pairs_vertex.size(); loop_index++)
			{
				const uint loop_vid = loop_pairs_vertex[loop_index].x;
				const uint adj_pair_idx = loop_pairs_vertex[loop_index].y;
				const uint loop_eid = loop_pair_ef_indices[loop_index].x;
				const uint loop_fid = loop_pair_ef_indices[loop_index].y;
				const uint new_loop_pair_idx = static_cast<uint>(ef_list.size());

				CollisionPair::EfPair pair;
				{
					const uint2	 loop_edge = luisa::make_uint2(loop_vid, loop_vid);
					const uint3	 loop_face = luisa::make_uint3(loop_vid, loop_vid, loop_vid);
					const float2 edge_bary = luisa::make_float2(1.0f, 0.0f);
					const float3 face_bary = luisa::make_float3(1.0f, 0.0f, 0.0f);
					pair.make_ef_pair(loop_edge, loop_face, edge_bary, face_bary);
					pair.set_gradient(luisa::make_float3(0.0f));
					pair.set_stiffness(0.0f);
				}
				ef_list.push_back(pair);
				ef_indices.push_back(luisa::make_uint2(loop_eid, loop_fid));
				ef_pair_adj_pairs_ext.push_back({ adj_pair_idx });

				ef_pair_adj_pairs_ext[adj_pair_idx].push_back(new_loop_pair_idx);
			}
		}

		num_pairs = static_cast<uint>(ef_indices.size());
		LUISA_INFO("EF pairs count = {} (= {} orig pairs + {} loop vertex pairs)",
			num_pairs,
			num_pairs_orig,
			loop_pairs_vertex.size());

		ef_pair_adj_pairs.resize(num_pairs, luisa::make_uint2(-1u, -1u));
		for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
		{
			const auto& adj_pairs_ext = ef_pair_adj_pairs_ext[pair_idx];
			uint2		adj_pairs = luisa::make_uint2(-1u, -1u);
			for (uint k = 0; k < luisa::min<uint>(2, static_cast<uint>(adj_pairs_ext.size())); k++)
				adj_pairs[k] = adj_pairs_ext[k];
			ef_pair_adj_pairs[pair_idx] = adj_pairs;
		}
	}
}; // namespace lcs

host_collision_data->narrow_phase_list_ef_indices = ef_indices;
host_collision_data->narrow_phase_list_ef = ef_list;
{
	host_collision_data->narrow_phase_collision_count[1] = num_pairs;
	Initializer::dynamic_resize_template(device, device_collision_data->narrow_phase_list_ef, num_pairs, "narrow_phase_list_ef (augmented)");
	stream << device_collision_data->narrow_phase_collision_count.view(1, 1).copy_from(&host_collision_data->narrow_phase_collision_count[1])
		   << device_collision_data->narrow_phase_list_ef.view(0, num_pairs).copy_from(host_collision_data->narrow_phase_list_ef.data())
		   << luisa::compute::synchronize();
}
host_untangling_data->loop_pairs_vertex = loop_pairs_vertex;
host_untangling_data->ef_pair_adj_pairs = ef_pair_adj_pairs;
host_untangling_data->ef_pair_adj_pairs_ext = ef_pair_adj_pairs_ext;

// Compute 2D and 3D positions of pairs
const auto&			sa_edges = host_mesh_data->sa_edges;
const auto&			sa_faces = host_mesh_data->sa_faces;
const auto&			sa_x = host_sim_data->sa_x;
const auto&			sa_material_x = host_mesh_data->sa_material_x;
std::vector<float4> ef_pair_pos_2D(num_pairs); // In edge mesh uv + face mesh uv
std::vector<float3> ef_pair_pos_3D(num_pairs);
CpuParallel::parallel_for(0,
	num_pairs,
	[&](const uint pair_idx)
	{
		const auto& ef_pair = ef_list[pair_idx];

		const uint2 edge = ef_pair.get_edge();
		const uint3 face = ef_pair.get_face();

		const float2 edge_bary = ef_pair.get_edge_bary();
		const float3 face_bary = ef_pair.get_face_bary();

		float2 uv_in_edge_mesh = sa_material_x[edge.x] * edge_bary.x
			+ sa_material_x[edge.y] * edge_bary.y;

		float2 uv_in_face_mesh = sa_material_x[face.x] * face_bary.x
			+ sa_material_x[face.y] * face_bary.y
			+ sa_material_x[face.z] * face_bary.z;
		float3 world_pos = sa_x[edge.x] * edge_bary.x + sa_x[edge.y] * edge_bary.y;

		float4 uvs = luisa::make_float4(uv_in_edge_mesh, uv_in_face_mesh);
		ef_pair_pos_2D[pair_idx] = uvs;
		ef_pair_pos_3D[pair_idx] = world_pos;
	});
host_untangling_data->ef_pair_pos_2D = ef_pair_pos_2D;
host_untangling_data->ef_pair_pos_3D = ef_pair_pos_3D;

// Compute pair area
std::vector<float> ef_pair_length(num_pairs, 0.f);
CpuParallel::parallel_for(0,
	num_pairs,
	[&](const uint pair_idx)
	{
		const auto& adj_pairs = ef_pair_adj_pairs_ext[pair_idx];
		float		sum_area = 0.0f;
		float3		curr_pos = ef_pair_pos_3D[pair_idx];
		for (const uint adj_pair_idx : adj_pairs)
		{
			float3 adj_world_pos = ef_pair_pos_3D[adj_pair_idx];
			sum_area += luisa::length(adj_world_pos - curr_pos);
		}
		ef_pair_length[pair_idx] = 0.5f * sum_area;
	});
host_untangling_data->ef_pair_length = ef_pair_length;
}

	float gia_host_hash_coord(const uint vid, const uint seed) noexcept
	{
		uint x = (vid ^ seed) * 0x27d4eb2du;
		x = x ^ (x >> 15u);
		x = x * 0x85ebca6bu;
		x = x ^ (x >> 13u);
		const float sgn = ((x & 1u) != 0u) ? 1.0f : -1.0f;
		const float mag = 0.35f + 0.65f * (static_cast<float>(x >> 1u) * (1.0f / 2147483647.0f));
		return sgn * mag;
	}

	float3 gia_host_perturb_pos(const float3& p, const uint vid, const float eps) noexcept
	{
		if (eps <= 0.0f)
			return p;
		return p + luisa::make_float3(
			gia_host_hash_coord(vid, 0x9e3779b9u),
			gia_host_hash_coord(vid, 0x7f4a7c15u),
			gia_host_hash_coord(vid, 0x6a09e667u)) * eps;
	}

ContourInfo identify_contour_type(const std::vector<std::vector<uint>>& pair_adj_pairs,
	const std::vector<luisa::ubyte>&									pair_is_boundary,
	const std::vector<uint>&											ef_pair_mesh_index,
	const std::vector<std::vector<uint>>&								intersection_contours,
	const uint															orig_num_pairs,
	const uint															contour_idx)
{
	auto is_loop_vertex = [orig_num_pairs](const uint pair_idx)
	{ return pair_idx >= orig_num_pairs; };

	const auto& contour = intersection_contours[contour_idx];

	std::vector<uint>		  contour_boundary_pairs;
	std::vector<uint>		  contour_loop_pairs;
	std::vector<luisa::ubyte> contour_loop_pairs_is_boundary;
	bool					  is_closed = !contour.empty();

	for (const uint pair_idx : intersection_contours[contour_idx])
	{
		if (is_loop_vertex(pair_idx))
		{
			contour_loop_pairs.push_back(pair_idx);
			contour_loop_pairs_is_boundary.push_back(0u);
			continue;
		}
		if (pair_is_boundary[pair_idx] != 0)
		{
			contour_boundary_pairs.push_back(pair_idx);
			is_closed = false;
		}
		else if (pair_adj_pairs[pair_idx].size() < 2u)
		{
			is_closed = false;
		}
	}

	ContourType contour_type = ContourType::Undifined;

	if (contour_boundary_pairs.empty())
	{
		if (is_closed)
		{
			if (contour_loop_pairs.empty())
				contour_type = ContourType::Closed;
			else if (contour_loop_pairs.size() == 1)
				contour_type = ContourType::Eight;
			else if (contour_loop_pairs.size() == 2)
				contour_type = ContourType::LL;
			else
				contour_type = ContourType::Undifined;
		}
		else
		{
			contour_type = ContourType::Undifined;
		}
	}
	else
	{
		if (contour_loop_pairs.empty())
		{
			if (contour_boundary_pairs.size() == 2)
			{
				const auto& front = contour_boundary_pairs.front();
				const auto& back = contour_boundary_pairs.back();
				contour_type = (ef_pair_mesh_index[front] == ef_pair_mesh_index[back]) ? ContourType::BBII : ContourType::BIBI;
			}
			else
			{
				contour_type = ContourType::Undifined;
			}
		}
		else if (contour_loop_pairs.size() == 1)
		{
			if (contour_boundary_pairs.size() == 1)
				contour_type = ContourType::BLI;
			else if (contour_boundary_pairs.size() == 2)
				contour_type = ContourType::Cross;
			else
				contour_type = ContourType::Undifined;
		}
	}

	std::array<bool, 2> mesh_is_closed = { false, false };
	switch (contour_type)
	{
		case ContourType::Closed:
		case ContourType::Eight:
		case ContourType::LL:
			mesh_is_closed = { true, true };
			break;
		case ContourType::BLI:
			mesh_is_closed = { false, false };
			break;
		case ContourType::Cross:
		{
			const uint front_mesh_idx = ef_pair_mesh_index[contour_boundary_pairs.front()];
			const uint back_mesh_idx = ef_pair_mesh_index[contour_boundary_pairs.back()];
			if (front_mesh_idx == back_mesh_idx)
				mesh_is_closed = (front_mesh_idx == 0) ? std::array<bool, 2>{ true, false } : std::array<bool, 2>{ false, true };
			else
				mesh_is_closed = { false, false };
			break;
		}
		case ContourType::BIBI:
			mesh_is_closed = { false, false };
			break;
		case ContourType::BBII:
		{
			const uint front_mesh_idx = ef_pair_mesh_index[contour_boundary_pairs.front()];
			mesh_is_closed = (front_mesh_idx == 0) ? std::array<bool, 2>{ true, false } : std::array<bool, 2>{ false, true };
			break;
		}
		case ContourType::Undifined:
			mesh_is_closed = { false, false };
			break;
	}

	return ContourInfo{
		.num_pairs = static_cast<uint>(contour.size()),
		.contour_type = contour_type,
		.loop_pairs = contour_loop_pairs,
		.boundary_pairs = contour_boundary_pairs,
		.mesh_is_closed = mesh_is_closed,
		.loop_pairs_is_boundary = contour_loop_pairs_is_boundary,
	};
}

void IntersectionResolver::host_identify_contour_type(Device& device, Stream& stream)
{
	auto& host_count = host_collision_data->narrow_phase_collision_count;
	uint& num_pairs = host_count[1];

	const uint num_loop_pairs = host_untangling_data->loop_pairs_vertex.size();
	const uint orig_num_pairs = num_pairs - num_loop_pairs;

	const auto& ef_pair_adj_pairs_ext = host_untangling_data->ef_pair_adj_pairs_ext;
	const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;
	const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;

	std::vector<std::vector<uint>>& intersection_contours = host_untangling_data->intersection_contours;

	std::vector<luisa::ubyte> pair_is_boundary(num_pairs, false);
	CpuParallel::parallel_for(0,
		num_pairs,
		[&](const uint pair_idx)
		{
			// Synthetic loop-vertex pairs must never be flagged as boundary edges.
			if (pair_idx >= orig_num_pairs || host_collision_data->narrow_phase_list_ef[pair_idx].get_is_loop_vertex())
			{
				pair_is_boundary[pair_idx] = 0u;
				return;
			}
			const uint eid = ef_indices[pair_idx].x;
			pair_is_boundary[pair_idx] = host_mesh_data->edge_adj_faces_ext[eid].size() <= 1 ? 1u : 0u;
		});

	if (host_untangling_data->num_contours.empty())
		host_untangling_data->num_contours.resize(1);
	uint& num_contours = host_untangling_data->num_contours.front();
	num_contours = intersection_contours.size();

	std::vector<ContourInfo>& intersection_contours_info = host_untangling_data->intersection_contours_info;
	intersection_contours_info.resize(num_contours);
	for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
	{
		auto contour_info = identify_contour_type(
			ef_pair_adj_pairs_ext, pair_is_boundary, ef_pair_mesh_index, intersection_contours, orig_num_pairs, contour_idx);
		intersection_contours_info[contour_idx] = contour_info;
	}

	std::array<uint, 8> contour_type_count = { 0 };
	for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
	{
		contour_type_count[static_cast<uint>(intersection_contours_info[contour_idx].contour_type)]++;
	}
	for (uint i = 0; i < contour_type_count.size(); i++)
	{
		if (contour_type_count[i] > 0)
		{
			LUISA_INFO("Contour type {:10} count: {}", contour_type_to_string(static_cast<ContourType>(i)), contour_type_count[i]);
		}
	}
}

void IntersectionResolver::host_identify_region(Device& device, Stream& stream)
{
	const auto& ef_list = host_collision_data->narrow_phase_list_ef;
	const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;
	const auto& ef_pair_edge_mesh_idx = host_untangling_data->ef_pair_mesh_index;
	const auto& ef_pair_contour_idx = host_untangling_data->ef_pair_contour_index;
	const uint	num_pairs = host_collision_data->narrow_phase_collision_count[1];
	const uint	num_faces = host_mesh_data->num_faces;
	const auto& sa_faces = host_mesh_data->sa_faces;
	const auto& sa_x = host_sim_data->sa_x;

	// 1. Propagate globally consistent face winding orientation across edge-adjacent manifold faces.
	gia_face_orient.assign(num_faces, 0);
	auto has_directed_edge = [](const uint3& f, const uint u, const uint v) noexcept -> bool
	{
		return (f.x == u && f.y == v) || (f.y == u && f.z == v) || (f.z == u && f.x == v);
	};
	std::queue<uint> face_queue;
	for (uint f0 = 0u; f0 < num_faces; ++f0)
	{
		if (gia_face_orient[f0] != 0)
			continue;
		gia_face_orient[f0] = 1;
		face_queue.push(f0);
		while (!face_queue.empty())
		{
			const uint	cf = face_queue.front();
			face_queue.pop();
			const uint3 f = sa_faces[cf];
			const uint3 f_edges = host_mesh_data->face_adj_edges[cf];
			for (uint k = 0u; k < 3u; ++k)
			{
				const uint	eid = f_edges[k];
				const auto& adj_faces = host_mesh_data->edge_adj_faces_ext[eid];
				if (adj_faces.size() != 2u)
					continue;
				const uint2 e = host_mesh_data->sa_edges[eid];
				const bool	cf_has_xy = has_directed_edge(f, e.x, e.y);
				for (const uint nf : adj_faces)
				{
					if (nf == cf || nf >= num_faces || gia_face_orient[nf] != 0)
						continue;
					const bool nf_has_xy = has_directed_edge(sa_faces[nf], e.x, e.y);
					// Adjacent faces with consistent orientation traverse their shared edge in opposite directions.
					gia_face_orient[nf] = (cf_has_xy == nf_has_xy) ? -gia_face_orient[cf] : gia_face_orient[cf];
					face_queue.push(nf);
				}
			}
		}
	}

	// 2. Compute oriented unit face normals (on unperturbed coordinates for response, and perturbed for sign consistency).
	gia_face_normals.resize(num_faces, luisa::make_float3(0.0f));
	CpuParallel::parallel_for(0u,
		num_faces,
		[&](const uint fid)
		{
			const uint3	 f = sa_faces[fid];
			const float3 n = luisa::cross(sa_x[f.y] - sa_x[f.x], sa_x[f.z] - sa_x[f.x]);
			const float	 len = luisa::length(n);
			gia_face_normals[fid] = len > 1e-12f
				? static_cast<float>(gia_face_orient[fid]) * (n / len)
				: luisa::make_float3(0.0f);
		});

	// 3. Assign consistent side parity per EF pair using signed dot product against the consistently oriented pierced face normal.
	std::vector<uint> ef_pair_region_idx(num_pairs, 0u);
	const float		  eps = gia_last_perturb_eps;
	CpuParallel::parallel_for(0u,
		num_pairs,
		[&](const uint pair_idx)
		{
			const uint contour_idx = ef_pair_contour_idx[pair_idx];
			const uint mesh_idx = ef_pair_edge_mesh_idx[pair_idx];
			uint	   edge_dir = 0u;
			if (!ef_list[pair_idx].get_is_loop_vertex())
			{
				const uint2	 edge = ef_list[pair_idx].get_edge();
				const uint	 fid = ef_indices[pair_idx].y;
				const uint3	 face = sa_faces[fid];
				const float3 e0_p = gia_host_perturb_pos(sa_x[edge.x], edge.x, eps);
				const float3 e1_p = gia_host_perturb_pos(sa_x[edge.y], edge.y, eps);
				const float3 a_p = gia_host_perturb_pos(sa_x[face.x], face.x, eps);
				const float3 b_p = gia_host_perturb_pos(sa_x[face.y], face.y, eps);
				const float3 c_p = gia_host_perturb_pos(sa_x[face.z], face.z, eps);
				float3		 n_p = luisa::cross(b_p - a_p, c_p - a_p);
				const float	 n_len = luisa::length(n_p);
				const float3 n_orient = n_len > 1e-12f
					? static_cast<float>(gia_face_orient[fid]) * (n_p / n_len)
					: gia_face_normals[fid];
				// edge_dir == 0 => edge.x lies on the +n_orient side of the pierced face; edge.y on the -n_orient side.
				const float signed_dot = luisa::dot(e0_p - e1_p, n_orient);
				edge_dir = (signed_dot >= 0.0f) ? 0u : 1u;
			}
			ef_pair_region_idx[pair_idx] = 4u * contour_idx + 2u * mesh_idx + edge_dir;
		});

	host_untangling_data->ef_pair_region_index = std::move(ef_pair_region_idx);
}

void IntersectionResolver::host_flood_filling(Device& device, Stream& stream)
{
	const auto& ef_list = host_collision_data->narrow_phase_list_ef;
	const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;
	const auto& intersection_contours = host_untangling_data->intersection_contours;
	const auto& intersection_contours_info = host_untangling_data->intersection_contours_info;
	const auto& ef_pair_region_idx = host_untangling_data->ef_pair_region_index;
	const auto& ef_pair_mesh_idx = host_untangling_data->ef_pair_mesh_index;

	const uint num_pairs = host_collision_data->narrow_phase_collision_count[1];
	const uint num_contours = static_cast<uint>(intersection_contours.size());
	const uint num_regions = num_contours * 4u;

	const uint num_verts = host_mesh_data->num_verts;
	const uint num_edges = host_mesh_data->num_edges;
	const uint num_faces = host_mesh_data->num_faces;

	std::vector<std::set<uint>>& region_contains_verts = host_untangling_data->region_contains_verts;
	region_contains_verts.clear();
	region_contains_verts.resize(num_regions);

	std::vector<std::set<uint>>& region_contains_verts_1order = host_untangling_data->region_contains_verts_1order;
	region_contains_verts_1order.clear();
	region_contains_verts_1order.resize(num_regions);

	gia_invalid_region_pair.assign(num_contours, luisa::make_uint2(-1u, -1u));
	gia_contour_pierced_faces.assign(num_contours, {});

	std::vector<float> region_rest_area(num_regions, 0.0f);
	std::vector<uint>  edge_cross_cnt(num_edges, 0u);
	std::vector<uint>  touched_edges;
	std::vector<int>   bfs_owner(num_verts, -1);
	std::vector<uint>  bfs_touched_verts;

	for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
	{
		const auto& contour = intersection_contours[contour_idx];
		const auto& contour_info = intersection_contours_info[contour_idx];

		// Collect loop vertices for this contour.
		std::unordered_set<uint> loop_vids;
		for (const uint lp_pair_idx : contour_info.loop_pairs)
		{
			if (lp_pair_idx >= num_pairs)
				continue;
			const uint2 edge = ef_list[lp_pair_idx].get_edge();
			if (edge.x == edge.y && edge.x < num_verts)
				loop_vids.insert(edge.x);
		}

		// Count real EF crossings per edge and collect pierced / incident faces per mesh side.
		touched_edges.clear();
		std::array<std::set<uint>, 2> pierced_face_sets;
		for (const uint pair_idx : contour)
		{
			if (ef_list[pair_idx].get_is_loop_vertex())
				continue;
			const uint eid = ef_indices[pair_idx].x;
			const uint fid = ef_indices[pair_idx].y;
			const uint m_e = ef_pair_mesh_idx[pair_idx];
			const uint m_f = m_e ^ 1u;

			if (edge_cross_cnt[eid] == 0u)
				touched_edges.push_back(eid);
			edge_cross_cnt[eid]++;

			if (fid < num_faces)
				pierced_face_sets[m_f].insert(fid);
			for (const uint adj_fid : host_mesh_data->edge_adj_faces_ext[eid])
			{
				if (adj_fid < num_faces)
					pierced_face_sets[m_e].insert(adj_fid);
			}
		}
		for (uint m = 0u; m < 2u; ++m)
		{
			gia_contour_pierced_faces[contour_idx][m].assign(
				pierced_face_sets[m].begin(), pierced_face_sets[m].end());
		}

		// Collect 1-ring seeds per mesh side using only odd-crossing edges (even crossings cancel topologically).
		std::array<std::array<std::set<uint>, 2>, 2> side_seeds;
		for (const uint pair_idx : contour)
		{
			if (ef_list[pair_idx].get_is_loop_vertex())
				continue;
			const uint eid = ef_indices[pair_idx].x;
			if ((edge_cross_cnt[eid] & 1u) == 0u)
				continue;

			const uint	m_e = ef_pair_mesh_idx[pair_idx];
			const uint	v0_parity = ef_pair_region_idx[pair_idx] & 1u;
			const uint	v1_parity = v0_parity ^ 1u;
			const uint2 edge = ef_list[pair_idx].get_edge();

			if (edge.x < num_verts && !loop_vids.count(edge.x))
				side_seeds[m_e][v0_parity].insert(edge.x);
			if (edge.y < num_verts && !loop_vids.count(edge.y))
				side_seeds[m_e][v1_parity].insert(edge.y);
		}
		for (uint m = 0u; m < 2u; ++m)
		{
			for (uint s = 0u; s < 2u; ++s)
				region_contains_verts_1order[4u * contour_idx + 2u * m + s] = side_seeds[m][s];
		}

		// Dual-BFS with leak detection for each mesh side.
		std::array<bool, 2> side_closed = { false, false };
		for (uint m = 0u; m < 2u; ++m)
		{
			const auto& s0 = side_seeds[m][0];
			const auto& s1 = side_seeds[m][1];
			// A closed separating wall must have non-empty seeds on BOTH sides; otherwise Dual-BFS
			// cannot detect a leak and would flood the entire valid mesh into a single region.
			if (s0.empty() || s1.empty())
				continue;

			bool seed_conflict = false;
			for (const uint v : s0)
			{
				if (s1.count(v))
				{
					seed_conflict = true;
					break;
				}
			}
			if (seed_conflict)
				continue;

			bfs_touched_verts.clear();
			std::array<std::queue<uint>, 2> queues;
			std::array<float, 2>			areas = { 0.0f, 0.0f };
			for (const uint v : s0)
			{
				bfs_owner[v] = 0;
				bfs_touched_verts.push_back(v);
				queues[0].push(v);
				areas[0] += host_mesh_data->sa_rest_vert_area[v];
			}
			for (const uint v : s1)
			{
				bfs_owner[v] = 1;
				bfs_touched_verts.push_back(v);
				queues[1].push(v);
				areas[1] += host_mesh_data->sa_rest_vert_area[v];
			}

			bool leaked = false;
			while ((!queues[0].empty() || !queues[1].empty()) && !leaked)
			{
				const uint side = (!queues[0].empty() && (queues[1].empty() || areas[0] <= areas[1])) ? 0u : 1u;
				const uint v = queues[side].front();
				queues[side].pop();

				const auto& adj_edges = host_mesh_data->vert_adj_edges[v];
				for (const uint eid : adj_edges)
				{
					// Odd-crossing contour edges form the flood-fill wall.
					if ((edge_cross_cnt[eid] & 1u) != 0u)
						continue;
					const uint2 edge = host_mesh_data->sa_edges[eid];
					const uint	nb = (edge.x == v) ? edge.y : edge.x;
					if (loop_vids.count(nb))
						continue;
					if (bfs_owner[nb] == -1)
					{
						bfs_owner[nb] = static_cast<int>(side);
						bfs_touched_verts.push_back(nb);
						queues[side].push(nb);
						areas[side] += host_mesh_data->sa_rest_vert_area[nb];
					}
					else if (bfs_owner[nb] != static_cast<int>(side))
					{
						leaked = true;
						break;
					}
				}
			}

			if (!leaked)
			{
				side_closed[m] = true;
				const uint r0 = 4u * contour_idx + 2u * m + 0u;
				const uint r1 = 4u * contour_idx + 2u * m + 1u;
				for (const uint v : bfs_touched_verts)
				{
					if (bfs_owner[v] == 0)
						region_contains_verts[r0].insert(v);
					else if (bfs_owner[v] == 1)
						region_contains_verts[r1].insert(v);
				}
				region_rest_area[r0] = areas[0];
				region_rest_area[r1] = areas[1];
			}

			for (const uint v : bfs_touched_verts)
				bfs_owner[v] = -1;
		}

		// Reset edge crossing counts for next contour.
		for (const uint eid : touched_edges)
			edge_cross_cnt[eid] = 0u;

		// Select invalid region(s) for this contour.
		auto pick_smaller_region = [&](const uint ra, const uint rb) -> uint
		{
			const uint na = static_cast<uint>(region_contains_verts[ra].size());
			const uint nb = static_cast<uint>(region_contains_verts[rb].size());
			if (na == 0u || nb == 0u)
				return -1u;
			const float aa = region_rest_area[ra];
			const float ab = region_rest_area[rb];
			if (std::abs(aa - ab) > 1e-7f * std::max(aa, ab))
				return (aa < ab) ? ra : rb;
			return (na <= nb) ? ra : rb;
		};

		uint2 invalid_regions = luisa::make_uint2(-1u, -1u);
		const uint r00 = 4u * contour_idx + 0u;
		const uint r01 = 4u * contour_idx + 1u;
		const uint r10 = 4u * contour_idx + 2u;
		const uint r11 = 4u * contour_idx + 3u;

		if (side_closed[0] && side_closed[1])
		{
			// On a single-sheet self-intersection (e.g. Eight, LL), m=0 and m=1 share one common connecting region.
			// Identify it so we pick the two disjoint penetrating flaps/caps instead of tagging the shared region.
			auto sets_overlap = [](const std::set<uint>& a, const std::set<uint>& b) -> bool
			{
				auto it_a = a.begin();
				auto it_b = b.begin();
				while (it_a != a.end() && it_b != b.end())
				{
					if (*it_a == *it_b)
						return true;
					if (*it_a < *it_b)
						++it_a;
					else
						++it_b;
				}
				return false;
			};

			int common_i0 = -1;
			int common_i1 = -1;
			for (uint i0 = 0u; i0 < 2u && common_i0 < 0; ++i0)
			{
				for (uint i1 = 0u; i1 < 2u; ++i1)
				{
					if (sets_overlap(region_contains_verts[4u * contour_idx + i0],
							region_contains_verts[4u * contour_idx + 2u + i1]))
					{
						common_i0 = static_cast<int>(i0);
						common_i1 = static_cast<int>(i1);
						break;
					}
				}
			}

			if (common_i0 >= 0)
			{
				const uint r_comm = 4u * contour_idx + static_cast<uint>(common_i0);
				const uint r0_excl = 4u * contour_idx + static_cast<uint>(common_i0 ^ 1);
				const uint r1_excl = 4u * contour_idx + 2u + static_cast<uint>(common_i1 ^ 1);
				const float a_comm = region_rest_area[r_comm];
				const float a0 = region_rest_area[r0_excl];
				const float a1 = region_rest_area[r1_excl];
				if (a_comm < std::min(a0, a1) && a_comm < 0.25f * (a0 + a1))
				{
					invalid_regions[0] = r_comm;
					invalid_regions[1] = -1u;
				}
				else
				{
					invalid_regions[0] = region_contains_verts[r0_excl].empty() ? -1u : r0_excl;
					invalid_regions[1] = region_contains_verts[r1_excl].empty() ? -1u : r1_excl;
				}
			}
			else
			{
				invalid_regions[0] = pick_smaller_region(r00, r01);
				invalid_regions[1] = pick_smaller_region(r10, r11);
			}
		}
		else if (side_closed[0])
		{
			invalid_regions[0] = pick_smaller_region(r00, r01);
		}
		else if (side_closed[1])
		{
			invalid_regions[1] = pick_smaller_region(r10, r11);
		}

		gia_invalid_region_pair[contour_idx] = invalid_regions;

		LUISA_INFO("Contour {} [{}] (flood_closed=[{},{}]): Invalid region pair {}, NumVerts = {} - {} (Compared to {} = {})",
			contour_idx,
			contour_type_to_string(contour_info.contour_type),
			side_closed[0],
			side_closed[1],
			invalid_regions,
			invalid_regions[0] != -1u ? region_contains_verts[invalid_regions[0]].size() : 0u,
			invalid_regions[1] != -1u ? region_contains_verts[invalid_regions[1]].size() : 0u,
			invalid_regions[0] != -1u ? region_contains_verts[invalid_regions[0] ^ 1u].size() : 0u,
			invalid_regions[1] != -1u ? region_contains_verts[invalid_regions[1] ^ 1u].size() : 0u);
	}

	// Build vertex / edge / face region CSR tags for proximity culling.
	std::vector<std::vector<uint>> vert_region_indices(num_verts);
	std::vector<std::vector<uint>> edge_region_indices(num_edges);
	std::vector<std::vector<uint>> face_region_indices(num_faces);

	for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
	{
		const uint2 invalid_regions = gia_invalid_region_pair[contour_idx];

		for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
		{
			const uint tag = 2u * contour_idx + mesh_idx;
			const uint invalid_region = invalid_regions[mesh_idx];
			if (invalid_region != -1u && invalid_region < region_contains_verts.size())
			{
				for (const uint vid : region_contains_verts[invalid_region])
					vert_region_indices[vid].push_back(tag);
			}
			// Always tag pierced faces directly so single-sided penetrations (0 vertices on opposite side) still disable opposing repulsion.
			for (const uint fid : gia_contour_pierced_faces[contour_idx][mesh_idx])
			{
				if (fid < num_faces)
					face_region_indices[fid].push_back(tag);
			}
		}

		// Disable conflicting self-collision repulsion directly across contour EF pairs on all contours
		// (matching PRP's contour-rim culling so valid-side 1-ring repulsion cannot lock invalid regions in place).
		for (const uint pair_idx : intersection_contours[contour_idx])
		{
			if (ef_list[pair_idx].get_is_loop_vertex())
				continue;
			const uint	m_e = ef_pair_mesh_idx[pair_idx];
			const uint	tag_e = 2u * contour_idx + m_e;
			const uint	tag_f = 2u * contour_idx + (m_e ^ 1u);
			const uint2 edge = ef_list[pair_idx].get_edge();
			const uint3 face = ef_list[pair_idx].get_face();
			vert_region_indices[edge.x].push_back(tag_e);
			vert_region_indices[edge.y].push_back(tag_e);
			vert_region_indices[face.x].push_back(tag_f);
			vert_region_indices[face.y].push_back(tag_f);
			vert_region_indices[face.z].push_back(tag_f);
		}
	}

	auto sort_unique = [](std::vector<uint>& vec)
	{
		std::sort(vec.begin(), vec.end());
		vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
	};
	for (auto& tags : vert_region_indices)
		sort_unique(tags);

	for (uint fid = 0u; fid < num_faces; ++fid)
	{
		const uint3 face = host_mesh_data->sa_faces[fid];
		auto&		vec = face_region_indices[fid];
		for (uint ii = 0u; ii < 3u; ++ii)
		{
			const auto& vtags = vert_region_indices[face[ii]];
			vec.insert(vec.end(), vtags.begin(), vtags.end());
		}
		sort_unique(vec);
	}
	for (uint eid = 0u; eid < num_edges; ++eid)
	{
		const uint2 edge = host_mesh_data->sa_edges[eid];
		auto&		vec = edge_region_indices[eid];
		for (uint ii = 0u; ii < 2u; ++ii)
		{
			const auto& vtags = vert_region_indices[edge[ii]];
			vec.insert(vec.end(), vtags.begin(), vtags.end());
		}
		sort_unique(vec);
	}

	Initializer::upload_2d_csr_from(host_untangling_data->vert_region_indices_csr, vert_region_indices);
	Initializer::upload_2d_csr_from(host_untangling_data->edge_region_indices_csr, edge_region_indices);
	Initializer::upload_2d_csr_from(host_untangling_data->face_region_indices_csr, face_region_indices);
}

void IntersectionResolver::host_resolve_ICM(Device& device, Stream& stream)
{
	const std::vector<uint>& host_count = host_collision_data->narrow_phase_collision_count;
	const uint				 num_pairs = host_count[1];

	std::vector<std::vector<uint>>& intersection_contours = host_untangling_data->intersection_contours;
	const uint						num_contours = intersection_contours.size();

	auto&		ef_list = host_collision_data->narrow_phase_list_ef;
	const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;
	const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;
	const auto& ef_pair_contour_index = host_untangling_data->ef_pair_contour_index;

	auto scale_gradient = [](const float3& G)
	{
		float  lenG = luisa::sqrt(luisa::dot(G, G));
		float3 direction = G / (lenG + 1e-10f);
		float  h0 = 0.01f;
		float  g0 = 1.f;
		float  H = h0 * lenG / luisa::sqrt(lenG * lenG + g0 * g0);
		return direction * H;
	};

	auto fn_need_inverse_grad = [&](const uint pair_idx)
	{
		const uint	contour_idx = ef_pair_contour_index[pair_idx];
		const auto& contour = intersection_contours[contour_idx];
		return ef_pair_mesh_index[pair_idx] != ef_pair_mesh_index[contour.front()];
	};

	CpuParallel::parallel_for(0,
		num_pairs,
		[&](const uint pair_idx)
		{
			const uint			   eid = ef_indices[pair_idx].x;
			const uint			   fid = ef_indices[pair_idx].y;
			CollisionPair::EfPair& info = ef_list[pair_idx];

			float3 local_grad = compute_local_gradient(eid,
				fid,
				info,
				host_mesh_data->edge_adj_faces_ext,
				host_mesh_data->sa_edges,
				host_mesh_data->sa_faces,
				host_sim_data->sa_x);

			local_grad = scale_gradient(local_grad);

			float kappa = 1e6f;
			float edge_area = host_mesh_data->sa_rest_edge_area[eid];
			float face_area = host_mesh_data->sa_rest_face_area[fid];
			float stiffness = 0.5f * (edge_area + face_area) * kappa;

			const float2 edge_bary = info.get_edge_bary();
			if (edge_bary[0] == 0.0f || edge_bary[1] == 0.0f)
			{
				stiffness = 0.0f;
				local_grad = luisa::make_float3(0.0f);
			}
			info.set_stiffness(stiffness);
			info.set_gradient(local_grad);
		});

	std::vector<float3> contour_gradients(num_contours, float3{ 0.0f, 0.0f, 0.0f });
	for (uint contour_idx = 0; contour_idx < num_contours; contour_idx++)
	{
		const auto& contour = intersection_contours[contour_idx];
		float3&		contour_gradient = contour_gradients[contour_idx];
		for (const uint pair_idx : contour)
		{
			const float3 local_grad = ef_list[pair_idx].get_gradient();
			const bool	 use_inverse = fn_need_inverse_grad(pair_idx);
			contour_gradient += use_inverse ? -local_grad : local_grad;
		}
		contour_gradient = scale_gradient(contour_gradient);
	}

	// Finalize stored ICM gradients using unified EF convention: cgB += -stiff * weights[ii] * orig_grad.
	// Assembly into cgB / cgA_diag is deferred to host_assemble_ef_response / device_assemble_ef_response.
	{
		CpuParallel::parallel_for(0,
			num_pairs,
			[&](const uint pair_idx)
			{
				const uint contour_idx = ef_pair_contour_index[pair_idx];
				auto&	   pair = ef_list[pair_idx];

				if (get_scene_params().accumulate_ICM_correction)
					pair.set_gradient(contour_gradients[contour_idx]);

				// Negate relative to old (+stiff * w * grad) convention so unified (-stiff * w * grad) produces identical force.
				const bool use_inverse = fn_need_inverse_grad(pair_idx);
				if (!use_inverse)
				{
					pair.flip_gradient();
				}

				const float3 orig_grad = pair.get_gradient();
				const float	 grad_norm_sqr = luisa::dot(orig_grad, orig_grad);
				if (!(grad_norm_sqr >= 1e-12f) || luisa::isinf(grad_norm_sqr) || luisa::isnan(grad_norm_sqr))
				{
					pair.set_stiffness(0.0f);
					pair.set_gradient(luisa::make_float3(0.0f, 1.0f, 0.0f));
				}
			});
	}
}

void IntersectionResolver::host_resolve_GIA(Device& device, Stream& stream)
{
	const std::vector<uint>& host_count = host_collision_data->narrow_phase_collision_count;
	const uint				 num_pairs = host_count[1];

	const auto& intersection_contours = host_untangling_data->intersection_contours;
	const uint	num_contours = static_cast<uint>(intersection_contours.size());
	const auto& intersection_contours_info = host_untangling_data->intersection_contours_info;
	const auto& region_contains_verts = host_untangling_data->region_contains_verts;

	auto&		ef_list = host_collision_data->narrow_phase_list_ef;
	const auto& ef_indices = host_collision_data->narrow_phase_list_ef_indices;
	const auto& ef_pair_mesh_index = host_untangling_data->ef_pair_mesh_index;
	const auto& ef_pair_contour_index = host_untangling_data->ef_pair_contour_index;
	const auto& sa_x = host_sim_data->sa_x;
	const auto& sa_faces = host_mesh_data->sa_faces;

	const uint	num_faces = host_mesh_data->num_faces;
	const float depth_threshold = get_scene_params().untangling_response_depth;
	// Match stiffness_untangling (same scale as stiffness_collision) so attraction is never overpowered by proximity repulsion.
	const float response_stiffness = get_scene_params().stiffness_untangling;
	// Cap per-step displacement relative to mesh edge length so CCD-free steps cannot overshoot and fold the cloth.
	const float safe_step_cap = std::min(
		depth_threshold,
		std::max(0.02f, (gia_cached_avg_edge_len > 0.0f ? 1.5f * gia_cached_avg_edge_len : 0.05f)));
	constexpr float response_area_floor = 3e-6f;
	constexpr float response_margin = 3e-3f;

	// Reset EF pair responses; only double-open contours populate active EF responses.
	gia_has_active_ef_response = false;
	for (uint pair_idx = 0u; pair_idx < num_pairs; ++pair_idx)
	{
		ef_list[pair_idx].set_stiffness(0.0f);
		ef_list[pair_idx].set_gradient(luisa::make_float3(0.0f, 1.0f, 0.0f));
	}

	std::vector<CollisionPair::CollisionPairTemplate> target_pairs;
	std::vector<uint4>								  target_indices;

	auto get_pair_edge_point = [&](const uint pair_idx) -> float3
	{
		const auto&	 pair = ef_list[pair_idx];
		const uint2	 edge = pair.get_edge();
		const float2 bary = pair.get_edge_bary();
		return sa_x[edge.x] * bary.x + sa_x[edge.y] * bary.y;
	};

	for (uint contour_idx = 0u; contour_idx < num_contours; ++contour_idx)
	{
		const auto& contour = intersection_contours[contour_idx];
		const auto& contour_info = intersection_contours_info[contour_idx];
		const uint2 invalid_regions = (contour_idx < gia_invalid_region_pair.size())
			? gia_invalid_region_pair[contour_idx]
			: luisa::make_uint2(-1u, -1u);

		const bool has_closed_region = (invalid_regions[0] != -1u || invalid_regions[1] != -1u);

		if (has_closed_region)
		{
			const bool	both_sides_pull = (invalid_regions[0] != -1u && invalid_regions[1] != -1u);
			const float side_scale = both_sides_pull ? 0.5f : 1.0f;

			// Case A: Global region-to-surface VF attraction for all vertices in closed invalid regions.
			for (uint mesh_idx = 0u; mesh_idx < 2u; ++mesh_idx)
			{
				const uint invalid_region = invalid_regions[mesh_idx];
				if (invalid_region == -1u || invalid_region >= region_contains_verts.size())
					continue;
				const auto& invalid_verts = region_contains_verts[invalid_region];
				if (invalid_verts.empty())
					continue;

				const uint opp_mesh = mesh_idx ^ 1u;
				const uint opp_invalid_region = invalid_regions[opp_mesh];

				// Build opposite target face set: pierced/contour-adjacent faces union faces touching opposite invalid region.
				std::set<uint> target_face_set;
				if (contour_idx < gia_contour_pierced_faces.size())
				{
					for (const uint fid : gia_contour_pierced_faces[contour_idx][opp_mesh])
					{
						if (fid < num_faces)
							target_face_set.insert(fid);
					}
				}
				if (opp_invalid_region != -1u && opp_invalid_region < region_contains_verts.size())
				{
					for (const uint ov : region_contains_verts[opp_invalid_region])
					{
						for (const uint fid : host_mesh_data->vert_adj_faces[ov])
						{
							if (fid < num_faces)
								target_face_set.insert(fid);
						}
					}
				}
				if (target_face_set.empty())
					continue;

				std::vector<uint> target_faces(target_face_set.begin(), target_face_set.end());
				std::vector<uint> src_verts(invalid_verts.begin(), invalid_verts.end());

				// Parity 0 means vertex lies on +N_f side of opposite faces => valid side is -N_f.
				const float valid_side_sign = ((invalid_region & 1u) == 0u) ? -1.0f : 1.0f;

				// Compute area-weighted coherent pull normal over opposite target faces to prevent inward crumpling
				// on faceted caps/pyramids (e.g. Closed, Eight, LL at high subdivision levels).
				float3 coherent_normal_sum = luisa::make_float3(0.0f);
				float  coherent_area_sum = 0.0f;
				for (const uint fid : target_faces)
				{
					const float	 af = std::max(host_mesh_data->sa_rest_face_area[fid], 1e-8f);
					const float3 nf = valid_side_sign * gia_face_normals[fid];
					if (luisa::dot(nf, nf) > 1e-12f)
					{
						coherent_normal_sum += af * nf;
						coherent_area_sum += af;
					}
				}
				const float	 coherent_len = luisa::length(coherent_normal_sum);
				const bool	 use_coherent_normal = (coherent_area_sum > 0.0f) && (coherent_len > 0.25f * coherent_area_sum);
				const float3 coherent_normal = use_coherent_normal
					? (coherent_normal_sum / coherent_len)
					: luisa::make_float3(0.0f);

				std::vector<CollisionPair::CollisionPairTemplate> local_pairs(src_verts.size());
				std::vector<uint4>								  local_indices(src_verts.size());
				std::vector<luisa::ubyte>						  local_valid(src_verts.size(), 0u);

				CpuParallel::parallel_for(0u,
					static_cast<uint>(src_verts.size()),
					[&](const uint idx)
					{
						const uint	 vid = src_verts[idx];
						const float3 xv = sa_x[vid];

						uint   best_fid = -1u;
						float  best_dist_sq = std::numeric_limits<float>::max();
						float3 best_bary = luisa::make_float3(1.0f / 3.0f);
						float3 best_q = xv;

						for (const uint fid : target_faces)
						{
							const uint3 f = sa_faces[fid];
							if (f.x == vid || f.y == vid || f.z == vid)
								continue;
							const float3 f0 = sa_x[f.x];
							const float3 f1 = sa_x[f.y];
							const float3 f2 = sa_x[f.z];
							const auto	 ebary = host_distance::point_triangle_distance_coeff_unclassified(
								Eigen::Vector3f(xv.x, xv.y, xv.z),
								Eigen::Vector3f(f0.x, f0.y, f0.z),
								Eigen::Vector3f(f1.x, f1.y, f1.z),
								Eigen::Vector3f(f2.x, f2.y, f2.z));
							const float3 bary = luisa::make_float3(ebary[0], ebary[1], ebary[2]);
							const float3 q = bary.x * f0 + bary.y * f1 + bary.z * f2;
							const float3 d = q - xv;
							const float	 d2 = luisa::dot(d, d);
							if (d2 < best_dist_sq)
							{
								best_dist_sq = d2;
								best_fid = fid;
								best_bary = bary;
								best_q = q;
							}
						}
						if (best_fid == -1u)
							return;

						const float3 diff = best_q - xv;
						const float	 dist = std::sqrt(best_dist_sq);

						float3 n_pull = luisa::make_float3(0.0f);
						float  pull_depth = 0.0f;
						if (use_coherent_normal)
						{
							n_pull = coherent_normal;
							pull_depth = std::max(0.0f, luisa::dot(diff, coherent_normal));
						}
						else
						{
							const float3 n_valid = valid_side_sign * gia_face_normals[best_fid];
							const float	 n_valid_len = luisa::length(n_valid);
							if (n_valid_len > 1e-6f)
							{
								const float3 n_valid_unit = n_valid / n_valid_len;
								const float	 signed_depth = luisa::dot(diff, n_valid_unit);
								if (dist > 1e-6f && luisa::dot(diff / dist, n_valid_unit) > 0.25f)
								{
									n_pull = diff / dist;
									pull_depth = dist;
								}
								else
								{
									n_pull = n_valid_unit;
									pull_depth = std::max(0.0f, signed_depth);
								}
							}
							else if (dist > 1e-6f)
							{
								n_pull = diff / dist;
								pull_depth = dist;
							}
							else
							{
								return;
							}
						}

						const float target_move = std::min(side_scale * (pull_depth + response_margin), safe_step_cap);
						const float C = -target_move;
						const float area = std::max(
							0.5f * (host_mesh_data->sa_rest_vert_area[vid] + host_mesh_data->sa_rest_face_area[best_fid]),
							response_area_floor);
						const float stiff = area * response_stiffness;
						const float k1 = stiff * C;
						const float k2 = stiff;
						if (!std::isfinite(k1) || !std::isfinite(k2) || !std::isfinite(n_pull.x) || !std::isfinite(n_pull.y) || !std::isfinite(n_pull.z))
							return;

						const uint3 face = sa_faces[best_fid];
						CollisionPair::CollisionPairTemplate pair;
						pair.make_vf_pair(luisa::make_uint4(vid, face.x, face.y, face.z), n_pull, k1, k2, area, best_bary);
						local_pairs[idx] = pair;
						local_indices[idx] = luisa::make_uint4(vid, best_fid, 2u * contour_idx + mesh_idx, 0u);
						local_valid[idx] = 1u;
					},
					32);

				for (size_t idx = 0; idx < src_verts.size(); ++idx)
				{
					if (local_valid[idx])
					{
						target_pairs.push_back(local_pairs[idx]);
						target_indices.push_back(local_indices[idx]);
					}
				}
			}
		}
		else if (get_scene_params().GIA_use_EF_response)
		{
			// Case B: Double-open contour (e.g. BIBI or leaked/open fallback) -> resolution-independent EF response.
			uint  n_real_pairs = 0u;
			float total_contour_len = 0.0f;
			for (const uint pair_idx : contour)
			{
				if (ef_list[pair_idx].get_is_loop_vertex())
					continue;
				n_real_pairs++;
				total_contour_len += host_untangling_data->ef_pair_length[pair_idx];
			}
			if (n_real_pairs == 0u)
				continue;

			if (contour_info.contour_type == ContourType::BIBI && contour_info.boundary_pairs.size() == 2u)
			{
				uint boundary_pair_0 = contour_info.boundary_pairs[0];
				uint boundary_pair_1 = contour_info.boundary_pairs[1];
				if (ef_pair_mesh_index[boundary_pair_0] == 1u && ef_pair_mesh_index[boundary_pair_1] == 0u)
					std::swap(boundary_pair_0, boundary_pair_1);

				const float3 boundary_point_0 = get_pair_edge_point(boundary_pair_0);
				const float3 boundary_point_1 = get_pair_edge_point(boundary_pair_1);
				const float3 boundary_vector = boundary_point_1 - boundary_point_0;
				const float	 boundary_len_sqr = luisa::dot(boundary_vector, boundary_vector);
				if (!(boundary_len_sqr > 1e-12f) || !std::isfinite(boundary_len_sqr))
					continue;

				const float	 boundary_len = std::sqrt(boundary_len_sqr);
				const float	 step_mag = std::min(safe_step_cap, boundary_len + response_margin);
				const float3 contour_grad = (boundary_vector / boundary_len) * step_mag;
				const float	 eff_area = std::max(boundary_len * boundary_len / static_cast<float>(n_real_pairs), response_area_floor);
				const float	 stiff = eff_area * response_stiffness;

				gia_has_active_ef_response = true;
				for (const uint pair_idx : contour)
				{
					if (ef_list[pair_idx].get_is_loop_vertex())
						continue;
					auto&	   pair = ef_list[pair_idx];
					const bool edge_on_mesh_0 = (ef_pair_mesh_index[pair_idx] == 0u);
					pair.set_stiffness(stiff);
					// Unified convention: cgB += -stiff * w * grad => mesh 0 moves along +contour_grad, mesh 1 along -contour_grad.
					pair.set_gradient(edge_on_mesh_0 ? -contour_grad : contour_grad);
				}
			}
			else if (contour_info.contour_type == ContourType::BLI
				&& contour_info.boundary_pairs.size() == 1u
				&& contour_info.loop_pairs.size() == 1u)
			{
				// Resolution-independent BLI separation: pull boundary mesh endpoint P_B toward loop vertex P_L
				// (direct geometric analogue of BIBI, avoiding 1/(E·N) grazing singularities in multi-pair ICM sums).
				const uint	 boundary_pair = contour_info.boundary_pairs[0];
				const uint	 loop_pair = contour_info.loop_pairs[0];
				const uint	 boundary_mesh = ef_pair_mesh_index[boundary_pair];
				const float3 boundary_point = get_pair_edge_point(boundary_pair);
				const float3 loop_point = get_pair_edge_point(loop_pair);
				const float3 sep_vec = loop_point - boundary_point;
				const float	 sep_len_sqr = luisa::dot(sep_vec, sep_vec);
				if (!(sep_len_sqr > 1e-12f) || !std::isfinite(sep_len_sqr))
					continue;

				const float	 sep_len = std::sqrt(sep_len_sqr);
				const float	 step_mag = std::min(safe_step_cap, sep_len + response_margin);
				const float3 contour_grad = (sep_vec / sep_len) * step_mag;
				const float	 eff_area = std::max(sep_len * sep_len / static_cast<float>(n_real_pairs), response_area_floor);
				const float	 stiff = eff_area * response_stiffness;

				gia_has_active_ef_response = true;
				for (const uint pair_idx : contour)
				{
					if (ef_list[pair_idx].get_is_loop_vertex())
						continue;
					auto&	   pair = ef_list[pair_idx];
					const bool edge_on_boundary_mesh = (ef_pair_mesh_index[pair_idx] == boundary_mesh);
					pair.set_stiffness(stiff);
					// Move boundary_mesh along +(P_L - P_B) and opposite side along -(P_L - P_B).
					pair.set_gradient(edge_on_boundary_mesh ? -contour_grad : contour_grad);
				}
			}
			else
			{
				// General open / leaked contour fallback: accumulated ICM contour-length minimization gradient.
				// Bound each pair's raw local gradient before summation so a single grazing pair (small E·N)
				// cannot dominate and destabilize the contour direction.
				constexpr uint ref_mesh_idx = 0u;
				float3		   accum_grad = luisa::make_float3(0.0f);
				for (const uint pair_idx : contour)
				{
					if (ef_list[pair_idx].get_is_loop_vertex())
						continue;
					const uint	 eid = ef_indices[pair_idx].x;
					const uint	 fid = ef_indices[pair_idx].y;
					const float3 raw_lg = compute_local_gradient(eid,
						fid,
						ef_list[pair_idx],
						host_mesh_data->edge_adj_faces_ext,
						host_mesh_data->sa_edges,
						host_mesh_data->sa_faces,
						sa_x);
					const float	 raw_len_sq = luisa::dot(raw_lg, raw_lg);
					if (!(raw_len_sq > 1e-12f) || !std::isfinite(raw_len_sq))
						continue;
					const float3 lg = raw_lg / std::sqrt(raw_len_sq + 1.0f);
					const bool	 inverse = (ef_pair_mesh_index[pair_idx] != ref_mesh_idx);
					accum_grad += inverse ? -lg : lg;
				}
				const float g_norm_sq = luisa::dot(accum_grad, accum_grad);
				if (!(g_norm_sq > 1e-12f) || !std::isfinite(g_norm_sq))
					continue;

				const float	 char_len = std::max(total_contour_len, 1e-3f);
				const float	 step_mag = std::min(safe_step_cap, 0.5f * char_len + response_margin);
				const float3 contour_grad = (accum_grad / std::sqrt(g_norm_sq)) * step_mag;
				const float	 eff_area = std::max(char_len * char_len / static_cast<float>(n_real_pairs), response_area_floor);
				const float	 stiff = eff_area * response_stiffness;

				gia_has_active_ef_response = true;
				for (const uint pair_idx : contour)
				{
					if (ef_list[pair_idx].get_is_loop_vertex())
						continue;
					auto&	   pair = ef_list[pair_idx];
					const bool inverse = (ef_pair_mesh_index[pair_idx] != ref_mesh_idx);
					pair.set_stiffness(stiff);
					// compute_local_gradient points in descent direction (+cgB), so negate for unified (-stiff * w * grad) convention.
					pair.set_gradient(inverse ? contour_grad : -contour_grad);
				}
			}
		}
	}

	// Upload VF attraction pairs into narrow_phase_list (automatically handled on both CPU and GPU PCG paths).
	if (!target_pairs.empty())
	{
		host_untangling_data->target_point_template_pairs = std::move(target_pairs);
		host_untangling_data->target_point_template_pairs_indices = std::move(target_indices);
		upload_response_pairs_to_gpu(device, stream, host_untangling_data, host_collision_data, device_collision_data);
	}
}

void IntersectionResolver::host_assemble_ef_response(Stream& stream, std::vector<float3>& sa_cgB, std::vector<float3x3>& sa_cgA_diag)
{
	const bool has_ef_response = get_scene_params().use_untangling_ICM
		|| (get_scene_params().use_untangling_GIA && get_scene_params().GIA_use_EF_response && gia_has_active_ef_response);
	const uint num_pairs = host_collision_data->narrow_phase_collision_count[1];
	if (!has_ef_response || num_pairs == 0u)
		return;

	const auto& ef_list = host_collision_data->narrow_phase_list_ef;
	CpuParallel::parallel_for(0u,
		num_pairs,
		[&](const uint pair_idx)
		{
			const auto&	 pair = ef_list[pair_idx];
			const float	 stiff = pair.get_stiffness();
			const float3 orig_grad = pair.get_gradient();
			const float	 grad_norm_sqr = luisa::dot(orig_grad, orig_grad);
			if (!(stiff > 0.0f) || !(grad_norm_sqr > 1e-12f) || luisa::isnan(stiff) || luisa::isinf(stiff))
				return;

			const float3   normal = orig_grad / std::sqrt(grad_norm_sqr);
			const float3x3 hess = stiff * outer_product(normal, normal);
			const auto	   indices = pair.get_indices();
			const auto	   weights = pair.get_weights();
			for (uint ii = 0u; ii < 5u; ++ii)
			{
				atomic_add(sa_cgB, -stiff * weights[ii] * orig_grad, indices[ii]);
				atomic_add(sa_cgA_diag, (weights[ii] * weights[ii]) * hess, indices[ii]);
			}
		});
}

void IntersectionResolver::device_assemble_ef_response(Stream& stream, Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag)
{
	const bool has_ef_response = get_scene_params().use_untangling_ICM
		|| (get_scene_params().use_untangling_GIA && get_scene_params().GIA_use_EF_response && gia_has_active_ef_response);
	const uint num_pairs = host_collision_data->narrow_phase_collision_count[1];
	if (!has_ef_response || num_pairs == 0u)
		return;

	stream << fn_assemble_ICM(get_collision_data(), sa_cgB, sa_cgA_diag).dispatch(num_pairs);
}

void IntersectionResolver::host_spmv_ICM(Stream& stream, const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr)
{
	auto&	   host_count = host_collision_data->narrow_phase_collision_count;
	const uint num_pairs = host_count[1];
	if (num_pairs == 0u)
		return;

	CpuParallel::parallel_for(0,
		num_pairs,
		[&](const uint pair_idx)
		{
			const auto&	 ef_pair = host_collision_data->narrow_phase_list_ef[pair_idx];
			const float	 stiff = ef_pair.get_stiffness();
			const float3 grad = ef_pair.get_gradient();
			const float	 len_sqr = luisa::dot(grad, grad);
			if (!(stiff > 0.0f) || !(len_sqr > 1e-12f))
				return;
			const float3 normal = grad / luisa::sqrt(len_sqr);

			const auto indices = ef_pair.get_indices();
			const auto weights = ef_pair.get_weights();

			float3 input_vec[5] = {
				input_ptr[indices[0]],
				input_ptr[indices[1]],
				input_ptr[indices[2]],
				input_ptr[indices[3]],
				input_ptr[indices[4]],
			};
			float3 output_vec[5] = {
				float3{ 0.0f, 0.0f, 0.0f },
				float3{ 0.0f, 0.0f, 0.0f },
				float3{ 0.0f, 0.0f, 0.0f },
				float3{ 0.0f, 0.0f, 0.0f },
				float3{ 0.0f, 0.0f, 0.0f },
			};
			float3x3 hess = stiff * outer_product(normal, normal);
			for (uint ii = 0; ii < 5; ii++)
			{
				for (uint jj = 0; jj < 5; jj++)
				{
					if (ii != jj)
					{
						output_vec[ii] += weights[ii] * weights[jj] * hess * input_vec[jj];
					}
				}
			}
			for (uint ii = 0; ii < 5; ii++)
			{
				atomic_add(output_ptr, output_vec[ii], indices[ii]);
			}
		});
}

void IntersectionResolver::host_spmv(Stream& stream, const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr)
{
	if (get_scene_params().use_untangling_ICM
		|| (get_scene_params().use_untangling_GIA && get_scene_params().GIA_use_EF_response && gia_has_active_ef_response))
	{
		host_spmv_ICM(stream, input_ptr, output_ptr);
	}
}

void IntersectionResolver::device_spmv_ICM(Stream& stream, const Buffer<float3>& input_ptr, Buffer<float3>& output_ptr)
{
	auto&	   host_count = host_collision_data->narrow_phase_collision_count;
	const uint num_pairs = host_count[1];
	if (num_pairs == 0u)
		return;
	stream << fn_spmv_ICM(get_collision_data(), input_ptr, output_ptr).dispatch(num_pairs);
}

void IntersectionResolver::device_spmv(Stream& stream, const Buffer<float3>& input_ptr, Buffer<float3>& output_ptr)
{
	if (get_scene_params().use_untangling_ICM
		|| (get_scene_params().use_untangling_GIA && get_scene_params().GIA_use_EF_response && gia_has_active_ef_response))
	{
		device_spmv_ICM(stream, input_ptr, output_ptr);
	}
}

void IntersectionResolver::host_resolve_PRP(Device& device, Stream& stream)
{
	luisa::Clock clock;
	host_resolve_intersections_PRP(device,
		stream,
		device_collision_data,
		host_collision_data,
		host_untangling_data,
		host_mesh_data,
		host_sim_data);

	if (get_scene_params().prp_debug)
	{
		auto& dbg = get_scene_params().prp_debug_info;
		dbg.float_stats["untangling_total_time"] = static_cast<float>(clock.toc());
	}
}

void IntersectionResolver::device_resolve_PRP(Device& device, Stream& stream)
{
	luisa::Clock clock;
	device_resolve_intersections_PRP(device,
		stream,
		device_collision_data,
		host_collision_data,
		device_untangling_data,
		host_untangling_data,
		device_mesh_data,
		host_mesh_data,
		device_sim_data,
		host_sim_data,
		lbvh_data_face,
		lbvh_data_edge,
		prp_shaders);

	if (get_scene_params().prp_debug)
	{
		auto& dbg = get_scene_params().prp_debug_info;
		dbg.float_stats["untangling_total_time"] = static_cast<float>(clock.toc());
	}
}

void IntersectionResolver::host_clear(Device& device, Stream& stream)
{
	if (!host_untangling_data->num_contours.empty())
		host_untangling_data->num_contours.front() = 0;
	host_untangling_data->intersection_contours.clear();
	host_untangling_data->intersection_contours_info.clear();
	host_untangling_data->target_point_template_pairs.clear();
	host_untangling_data->target_point_template_pairs_indices.clear();
	host_untangling_data->vert_region_indices_csr.clear();
	host_untangling_data->edge_region_indices_csr.clear();
	host_untangling_data->face_region_indices_csr.clear();
	gia_invalid_region_pair.clear();
	gia_contour_pierced_faces.clear();
	gia_has_active_ef_response = false;
}

void IntersectionResolver::host_resolve(Device& device, Stream& stream)
{
	get_scene_params().validate_untangling_configuration();

	auto& host_count = host_collision_data->narrow_phase_collision_count;
	uint& num_pairs = host_count[1];
	auto  publish_input_ef_pair_count = [&]()
	{
		if (get_scene_params().prp_debug || get_scene_params().collect_iteration_debug)
			get_scene_params().prp_debug_info.uint_stats["intersection_resolver_input_ef_pair_count"] = num_pairs;
	};
	publish_input_ef_pair_count();

	if (num_pairs != 0)
		LUISA_INFO("Intersection pairs count: {}", num_pairs);

	host_clear(device, stream);

	if (num_pairs == 0)
	{
		return;
	}
	if (get_scene_params().use_untangling_GIA)
		get_scene_params().use_ccd_linesearch = false;

	download_intersection_list(stream);

	host_preprocess_ef_pairs(device, stream);

	num_pairs = host_count[1];
	if (num_pairs == 0u)
	{
		publish_input_ef_pair_count();
		return;
	}

	if (get_scene_params().use_untangling_PRP)
	{
		if (get_scene_params().use_gpu_untangling)
		{
			device_resolve_PRP(device, stream);
		}
		else
		{
			host_resolve_PRP(device, stream);
		}
		publish_input_ef_pair_count();
	}
	else
	{
		if (get_scene_params().use_untangling_ICM)
		{
			intersection_contour_construction_from_ext_adjacent(host_untangling_data, host_collision_data);
			host_resolve_ICM(device, stream);
		}
		else if (get_scene_params().use_untangling_GIA)
		{
			intersection_contour_construction_from_ext_adjacent(host_untangling_data, host_collision_data);
			host_identify_contour_type(device, stream);
			host_identify_region(device, stream);
			host_flood_filling(device, stream);
			host_resolve_GIA(device, stream);
		}
	}

	if (!host_untangling_data->vert_region_indices_csr.empty())
	{
		const uint size_vert = host_untangling_data->vert_region_indices_csr.size();
		Initializer::dynamic_resize_template(device, device_untangling_data->vert_region_indices_csr, size_vert, "vert_region_indices_csr");
		stream << device_untangling_data->vert_region_indices_csr.view(0, size_vert).copy_from(host_untangling_data->vert_region_indices_csr.data());
	}
	if (!host_untangling_data->edge_region_indices_csr.empty())
	{
		const uint size_edge = host_untangling_data->edge_region_indices_csr.size();
		const uint size_face = host_untangling_data->face_region_indices_csr.size();
		Initializer::dynamic_resize_template(device, device_untangling_data->edge_region_indices_csr, size_edge, "edge_region_indices_csr");
		Initializer::dynamic_resize_template(device, device_untangling_data->face_region_indices_csr, size_face, "face_region_indices_csr");
		stream << device_untangling_data->edge_region_indices_csr.view(0, size_edge).copy_from(host_untangling_data->edge_region_indices_csr.data());
		stream << device_untangling_data->face_region_indices_csr.view(0, size_face).copy_from(host_untangling_data->face_region_indices_csr.data());
	}

	Initializer::dynamic_resize_template(device, device_collision_data->narrow_phase_list_ef, num_pairs, "narrow_phase_list_ef region property");

	stream << device_collision_data->narrow_phase_collision_count.view(1, 1).copy_from(&num_pairs)
		   << device_collision_data->narrow_phase_list_ef.view(0, num_pairs).copy_from(host_collision_data->narrow_phase_list_ef.data());
	stream << luisa::compute::synchronize();
}
} // namespace lcs

namespace lcs
{

	enum class RegionMatchTypeInput : uint
	{
		VF = 0,
		EE = 1,
		Dynamic = 2,
	};

	// PairType: 0 for VF, 1 for EE, 2 for dynamic
	template <RegionMatchTypeInput PairType>
	luisa::compute::Uint fn_is_matched_invalid_region(
		luisa::compute::BufferVar<uint>& vert_region_indices_csr,
		luisa::compute::BufferVar<uint>& edge_region_indices_csr,
		luisa::compute::BufferVar<uint>& face_region_indices_csr,
		const luisa::compute::Uint2&	 indices,
		const luisa::compute::Uint		 type)
	{
		using namespace luisa::compute;
		auto traverse_indices = [](BufferVar<uint>&	 left_region_indices_csr,
									BufferVar<uint>& right_region_indices_csr,
									const Uint2&	 indices,
									const Uint2&	 prefix,
									const Uint2&	 adj_count) -> Bool
		{
			Bool is_matched = false;
			{
				const Uint left = indices[0];
				const Uint left_count = adj_count[0];

				const Uint right = indices[1];
				const Uint right_count = adj_count[1];
				$if(left_count > 0u & right_count > 0u)
				{
					$for(li, left_count)
					{
						const Uint left_region_idx = left_region_indices_csr.read(prefix[0] + li);
						$for(rj, right_count)
						{
							const Uint right_region_idx = right_region_indices_csr.read(prefix[1] + rj);
							$if(left_region_idx == (right_region_idx ^ 1))
							{
								is_matched = true;
								$break;
							};
						};
					};
				};
			}
			return is_matched;
		};

		Uint2 prefix;
		Uint2 suffix;
		Uint2 adj_count;

		auto get_vf_prefix_suffix = [&]()
		{
			prefix = make_uint2(vert_region_indices_csr.read(indices[0]), face_region_indices_csr.read(indices[1]));
			suffix = make_uint2(vert_region_indices_csr.read(indices[0] + 1), face_region_indices_csr.read(indices[1] + 1));
		};
		auto get_ee_prefix_suffix = [&]()
		{
			prefix = make_uint2(edge_region_indices_csr.read(indices[0]), edge_region_indices_csr.read(indices[1]));
			suffix = make_uint2(edge_region_indices_csr.read(indices[0] + 1), edge_region_indices_csr.read(indices[1] + 1));
		};

		if constexpr (PairType == RegionMatchTypeInput::VF)
		{
			get_vf_prefix_suffix();
		}
		else if constexpr (PairType == RegionMatchTypeInput::EE)
		{
			get_ee_prefix_suffix();
		}
		else
		{
			$if(type == CollisionPair::type_vf())
			{
				get_vf_prefix_suffix();
			}
			$elif(type == CollisionPair::type_ee())
			{
				get_ee_prefix_suffix();
			};
		}

		adj_count = suffix - prefix;

		Uint result = 0u;
		$if(any(adj_count != 0u))
		{
			Bool is_matched = false;

			if constexpr (PairType == RegionMatchTypeInput::VF)
			{
				is_matched = traverse_indices(vert_region_indices_csr, face_region_indices_csr, indices, prefix, adj_count);
			}
			else if constexpr (PairType == RegionMatchTypeInput::EE)
			{
				is_matched = traverse_indices(edge_region_indices_csr, edge_region_indices_csr, indices, prefix, adj_count);
			}
			else
			{
				$if(type == CollisionPair::type_vf())
				{
					is_matched = traverse_indices(vert_region_indices_csr, face_region_indices_csr, indices, prefix, adj_count);
				}
				$elif(type == CollisionPair::type_ee())
				{
					is_matched = traverse_indices(edge_region_indices_csr, edge_region_indices_csr, indices, prefix, adj_count);
				};
			}

			$if(is_matched)
			{
				result = 1u;
			}
			$else
			{
				result = 2u;
			};
		};
		return result;
	};

	void IntersectionResolver::compile_resolve(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;
		// Set invalid vertices from proximity pairs
		compiler.compile<1>(
			fn_identify_invalid_verts_from_proximity_pairs,
			[](Var<CDBG> collision_data, BufferVar<float3> sa_x, BufferVar<float> sa_per_vert_d_hat, BufferVar<float> sa_per_vert_offset)
			{
				auto& narrow_phase_list = collision_data.narrow_phase_list;

				const Uint pair_idx = dispatch_id().x;
				auto	   pair = narrow_phase_list.read(pair_idx);
				auto	   indices = pair->get_indices();
				auto	   weight = pair->get_weight();
				Float3	   delta = weight[0] * sa_x.read(indices[0]) + weight[1] * sa_x.read(indices[1])
					+ weight[2] * sa_x.read(indices[2]) + weight[3] * sa_x.read(indices[3]);
				Float d2 = dot(delta, delta);
				Float thickness = sa_per_vert_offset.read(indices[0]) + sa_per_vert_offset.read(indices[2]);
				$if(d2 < thickness * thickness + 1e-10f) // thickness + 1e-5
				{
					auto& vert_is_invalid = collision_data.vert_is_invalid;
					for (uint ii = 0; ii < 4; ii++)
					{
						vert_is_invalid.write(indices[ii], 1u);
					}
				};
			});
		compiler.compile<1>(fn_filter_invalid_proximity_pairs_from_invalid_verts,
			[](Var<CDBG> collision_data)
			{
				auto& narrow_phase_list = collision_data.narrow_phase_list;
				auto& vert_is_invalid = collision_data.vert_is_invalid;

				const Uint pair_idx = dispatch_id().x;
				auto	   pair = collision_data.narrow_phase_list.read(pair_idx);
				auto	   indices = pair->get_indices();
				Bool	   is_invalid = false;
				for (uint ii = 0; ii < 4; ii++)
				{
					$if(vert_is_invalid.read(indices[ii]) != 0u)
					{
						is_invalid = true;
					};
				}

				// If invalid vertex present -> invalidate this proximity pair (existing behavior)
				$if(is_invalid)
				{
					pair->disable();
					narrow_phase_list.write(pair_idx, pair);
					$return();
				};
			});

		compiler.compile<1>(fn_filter_invalid_proximity_pairs_from_region_info,
			[](BufferVar<CollisionPair::CollisionPairTemplate> narrow_phase_list,
				BufferVar<uint2>							   narrow_phase_list_indices,
				BufferVar<uint>								   vert_region_indices_csr,
				BufferVar<uint>								   edge_region_indices_csr,
				BufferVar<uint>								   face_region_indices_csr)
			{
				const Uint pair_idx = dispatch_id().x;
				auto	   pair = narrow_phase_list.read(pair_idx);
				auto	   indices = narrow_phase_list_indices.read(pair_idx);
				auto	   type = pair->get_collision_type();
				$if(fn_is_matched_invalid_region<RegionMatchTypeInput::Dynamic>(vert_region_indices_csr, edge_region_indices_csr, face_region_indices_csr, indices, type) == 1)
				{
					pair->disable();
					narrow_phase_list.write(pair_idx, pair);
					$return();
				};
			});

		constexpr bool cull_adj_pairs = false;
		constexpr bool cull_near_contour_pairs = true;
		constexpr uint flag_vert_is_region_match = 1 << 31;
		constexpr uint flag_vert_is_near_contour = 1 << 30;

		// For CCD culling
		compiler.compile<1>(fn_filter_broadphase_vf_pairs,
			[flag_vert_is_region_match, flag_vert_is_near_contour](
				BufferVar<uint>	 broadphase_list,
				BufferVar<uint>	 vert_is_invalid,
				BufferVar<uint>	 vert_region_indices_csr,
				BufferVar<uint>	 edge_region_indices_csr,
				BufferVar<uint>	 face_region_indices_csr,
				BufferVar<uint3> sa_faces,
				const Uint		 dispatch_prefix)
			{
				const Uint pair_idx = dispatch_x() + dispatch_prefix;

				Uint		vid = broadphase_list.read(2 * pair_idx + 0);
				Uint		fid = broadphase_list.read(2 * pair_idx + 1);
				const Uint3 face = sa_faces.read(fid);
				const Uint	type = CollisionPair::type_vf();
				const Uint4 vert_indices = make_uint4(vid, face.x, face.y, face.z);
				const Uint2 vf_indices = make_uint2(vid, fid);

				Bool is_adj = false;
				if constexpr (cull_adj_pairs)
					is_adj = (vid == face.x) | (vid == face.y) | (vid == face.z);

				Bool is_invalid = false;
				if constexpr (cull_near_contour_pairs)
					is_invalid =
						vert_is_invalid.read(vert_indices[0]) != 0u | //
						vert_is_invalid.read(vert_indices[1]) != 0u | //
						vert_is_invalid.read(vert_indices[2]) != 0u | //
						vert_is_invalid.read(vert_indices[3]) != 0u;

				Bool is_region_mismatch = false;
				is_region_mismatch = fn_is_matched_invalid_region<RegionMatchTypeInput::VF>(vert_region_indices_csr, edge_region_indices_csr, face_region_indices_csr, vf_indices, type) == 1;

				$if(is_adj | is_invalid | is_region_mismatch)
				{
					$if(is_region_mismatch)
					{
						vid |= flag_vert_is_region_match;
						fid |= flag_vert_is_region_match;
					};
					if constexpr (cull_near_contour_pairs)
					{
						$if(is_invalid)
						{
							vid |= flag_vert_is_near_contour;
							fid |= flag_vert_is_near_contour;
						};
					}
					broadphase_list.write(2 * pair_idx + 0, vid);
					broadphase_list.write(2 * pair_idx + 1, fid);
				};
			});

		compiler.compile<1>(fn_filter_broadphase_ee_pairs,
			[flag_vert_is_region_match, flag_vert_is_near_contour](
				BufferVar<uint>	 broadphase_list,
				BufferVar<uint>	 vert_is_invalid,
				BufferVar<uint>	 vert_region_indices_csr,
				BufferVar<uint>	 edge_region_indices_csr,
				BufferVar<uint>	 face_region_indices_csr,
				BufferVar<uint2> sa_edges,
				const Uint		 dispatch_prefix)
			{
				const Uint pair_idx = dispatch_x() + dispatch_prefix;

				Uint		eid1 = broadphase_list.read(2 * pair_idx + 0);
				Uint		eid2 = broadphase_list.read(2 * pair_idx + 1);
				const Uint2 edge1 = sa_edges.read(eid1);
				const Uint2 edge2 = sa_edges.read(eid2);
				const Uint	type = CollisionPair::type_ee();

				const Uint4 vert_indices = make_uint4(edge1.x, edge1.y, edge2.x, edge2.y);
				const Uint2 ee_indices = make_uint2(eid1, eid2);

				Bool is_adj = false;
				if constexpr (cull_adj_pairs)
					is_adj = (edge1.x == edge2.x) | (edge1.x == edge2.y) | (edge1.y == edge2.x) | (edge1.y == edge2.y);

				Bool is_invalid = false;
				if constexpr (cull_near_contour_pairs)
					is_invalid =
						vert_is_invalid.read(vert_indices[0]) != 0u | //
						vert_is_invalid.read(vert_indices[1]) != 0u | //
						vert_is_invalid.read(vert_indices[2]) != 0u | //
						vert_is_invalid.read(vert_indices[3]) != 0u;

				Bool is_region_mismatch = false;
				is_region_mismatch = fn_is_matched_invalid_region<RegionMatchTypeInput::EE>(vert_region_indices_csr, edge_region_indices_csr, face_region_indices_csr, ee_indices, type) == 1;

				$if(is_adj | is_invalid | is_region_mismatch)
				{
					$if(is_region_mismatch)
					{
						eid1 |= flag_vert_is_region_match;
						eid2 |= flag_vert_is_region_match;
					};
					if constexpr (cull_near_contour_pairs)
					{
						$if(is_invalid)
						{
							eid1 |= flag_vert_is_near_contour;
							eid2 |= flag_vert_is_near_contour;
						};
					}
					broadphase_list.write(2 * pair_idx + 0, eid1);
					broadphase_list.write(2 * pair_idx + 1, eid2);
				};
			});

		compiler.compile<1>(fn_spmv_ICM,
			[](Var<CDBG> collision_data, BufferVar<float3> input_ptr, BufferVar<float3> output_ptr)
			{
				const Uint	 pair_idx = dispatch_x();
				const auto	 ef_pair = collision_data.narrow_phase_list_ef.read(pair_idx);
				const Float3 grad = ef_pair->get_gradient();
				const Float	 stiff = ef_pair->get_stiffness();
				const Float	 len_sqr = dot(grad, grad);
				$if(stiff <= 0.0f | len_sqr <= 1e-12f)
				{
					$return();
				};
				const Float3 normal = grad / sqrt(len_sqr);

				const auto indices = ef_pair->get_indices();
				const auto weights = ef_pair->get_weights();

				Float3 input_vec[5] = {
					input_ptr.read(indices[0]),
					input_ptr.read(indices[1]),
					input_ptr.read(indices[2]),
					input_ptr.read(indices[3]),
					input_ptr.read(indices[4]),
				};
				Float3 output_vec[5] = {
					float3{ 0.0f, 0.0f, 0.0f },
					float3{ 0.0f, 0.0f, 0.0f },
					float3{ 0.0f, 0.0f, 0.0f },
					float3{ 0.0f, 0.0f, 0.0f },
					float3{ 0.0f, 0.0f, 0.0f },
				};
				for (uint ii = 0; ii < 5; ii++)
				{
					for (uint jj = 0; jj < 5; jj++)
					{
						if (ii != jj)
						{
							output_vec[ii] += (stiff * weights[ii] * weights[jj]) * (outer_product(normal, normal) * input_vec[jj]);
						}
					}
				}
				for (uint ii = 0; ii < 5; ii++)
				{
					output_ptr.atomic(indices[ii])[0].fetch_add(output_vec[ii][0]);
					output_ptr.atomic(indices[ii])[1].fetch_add(output_vec[ii][1]);
					output_ptr.atomic(indices[ii])[2].fetch_add(output_vec[ii][2]);
				}
			});

		compiler.compile<1>(fn_assemble_ICM,
			[](Var<CDBG> collision_data, BufferVar<float3> sa_cgB, BufferVar<float3x3> sa_cgA_diag)
			{
				const Uint	 pair_idx = dispatch_x();
				const auto	 ef_pair = collision_data.narrow_phase_list_ef.read(pair_idx);
				const Float	 stiff = ef_pair->get_stiffness();
				const Float3 orig_grad = ef_pair->get_gradient();
				const Float	 len_sqr = dot(orig_grad, orig_grad);
				$if(stiff <= 0.0f | len_sqr <= 1e-12f)
				{
					$return();
				};
				const Float3   normal = orig_grad / sqrt(len_sqr);
				const auto	   indices = ef_pair->get_indices();
				const auto	   weights = ef_pair->get_weights();
				const Float3x3 hess = stiff * outer_product(normal, normal);
				for (uint ii = 0; ii < 5; ii++)
				{
					const Float3   rhs = -stiff * weights[ii] * orig_grad;
					const Float3x3 diag_block = (weights[ii] * weights[ii]) * hess;
					const Uint	   vid = indices[ii];
					sa_cgB.atomic(vid)[0].fetch_add(rhs[0]);
					sa_cgB.atomic(vid)[1].fetch_add(rhs[1]);
					sa_cgB.atomic(vid)[2].fetch_add(rhs[2]);
					sa_cgA_diag.atomic(vid)[0][0].fetch_add(diag_block[0][0]);
					sa_cgA_diag.atomic(vid)[0][1].fetch_add(diag_block[0][1]);
					sa_cgA_diag.atomic(vid)[0][2].fetch_add(diag_block[0][2]);
					sa_cgA_diag.atomic(vid)[1][0].fetch_add(diag_block[1][0]);
					sa_cgA_diag.atomic(vid)[1][1].fetch_add(diag_block[1][1]);
					sa_cgA_diag.atomic(vid)[1][2].fetch_add(diag_block[1][2]);
					sa_cgA_diag.atomic(vid)[2][0].fetch_add(diag_block[2][0]);
					sa_cgA_diag.atomic(vid)[2][1].fetch_add(diag_block[2][1]);
					sa_cgA_diag.atomic(vid)[2][2].fetch_add(diag_block[2][2]);
				}
			});
	}
	void dispatch_large_thread_template(const std::function<void(uint, uint)>& dispatch_func, uint total_size)
	{
		constexpr uint max_threads_per_dispatch = 65535 * 256;
		for (uint loop = 0; loop < get_dispatch_block(total_size, max_threads_per_dispatch); ++loop)
		{
			const uint dispatch_prefix = loop * max_threads_per_dispatch;
			const uint curr_dispatch_size = min_scalar(total_size - dispatch_prefix, max_threads_per_dispatch);
			dispatch_func(curr_dispatch_size, dispatch_prefix);
		}
	}

	void IntersectionResolver::filter_broad_phase_pairs(Stream& stream)
	{
		auto&	   host_broad_count = host_collision_data->broad_phase_collision_count;
		const uint num_vf_broad = host_broad_count[CollisionPair::CollisionCount::vf_offset()];
		const uint num_ee_broad = host_broad_count[CollisionPair::CollisionCount::ee_offset()];

		auto&	   host_narrow_count = host_collision_data->narrow_phase_collision_count;
		const uint num_ef = host_narrow_count[1];

		// For CCD
		if (num_ef != 0 && !host_untangling_data->vert_region_indices_csr.empty())
		{
			if (num_vf_broad != 0)
			{
				dispatch_large_thread_template(
					[&](uint curr_dispatch_size, uint dispatch_prefix)
					{
						stream << fn_filter_broadphase_vf_pairs(
							device_collision_data->broad_phase_list_vf,
							device_collision_data->vert_is_invalid,
							device_untangling_data->vert_region_indices_csr,
							device_untangling_data->edge_region_indices_csr,
							device_untangling_data->face_region_indices_csr,
							device_mesh_data->sa_faces, dispatch_prefix)
									  .dispatch(curr_dispatch_size);
					},
					num_vf_broad);
			}
			if (num_ee_broad != 0)
			{
				dispatch_large_thread_template(
					[&](uint curr_dispatch_size, uint dispatch_prefix)
					{
						stream << fn_filter_broadphase_ee_pairs(
							device_collision_data->broad_phase_list_ee,
							device_collision_data->vert_is_invalid,
							device_untangling_data->vert_region_indices_csr,
							device_untangling_data->edge_region_indices_csr,
							device_untangling_data->face_region_indices_csr,
							device_mesh_data->sa_edges, dispatch_prefix)
									  .dispatch(curr_dispatch_size);
					},
					num_ee_broad);
			}
		}
	}
	void IntersectionResolver::filter_invalid_pairs(Stream& stream)
	{
		auto&	   host_count = host_collision_data->narrow_phase_collision_count;
		const uint num_vf_ee = host_count.front();
		const uint num_ef = host_count[1];
		const uint num_prp_response_pairs = host_untangling_data->target_point_template_pairs.size();
		if (num_prp_response_pairs > num_vf_ee)
		{
			LUISA_ERROR("PRP response pair count {} exceeds narrow-phase pair count {}.",
				num_prp_response_pairs,
				num_vf_ee);
		}
		const uint num_proximity_pairs = num_vf_ee - num_prp_response_pairs;

		if (num_ef != 0)
		{
			// Identify invalid vertices
			if (num_proximity_pairs != 0u)
			{
				stream << fn_identify_invalid_verts_from_proximity_pairs(get_collision_data(),
					device_sim_data->sa_x,
					device_sim_data->sa_contact_active_verts_d_hat,
					device_sim_data->sa_contact_active_verts_offset)
							  .dispatch(num_proximity_pairs);
			}
			// Disable proximity pairs near contours (if any vertex is invalid, disable the pair)
			if (num_proximity_pairs != 0u)
			{
				stream << fn_filter_invalid_proximity_pairs_from_invalid_verts(get_collision_data()).dispatch(num_proximity_pairs);

				if (!host_untangling_data->vert_region_indices_csr.empty() && get_scene_params().use_untangling)
				{
					stream << fn_filter_invalid_proximity_pairs_from_region_info(
						device_collision_data->narrow_phase_list,
						device_collision_data->narrow_phase_list_indices,
						device_untangling_data->vert_region_indices_csr,
						device_untangling_data->edge_region_indices_csr,
						device_untangling_data->face_region_indices_csr)
								  .dispatch(num_proximity_pairs);
				}
			}
		}
	}
	void IntersectionResolver::get_intersection_curves(
		std::vector<std::array<float, 3>>& output_positions,
		std::vector<std::array<uint, 2>>&  output_edges,
		std::vector<std::array<float, 3>>& output_colors)
	{
		get_intersection_curve_data(
			output_positions,
			output_edges,
			output_colors,
			host_untangling_data,
			host_collision_data,
			host_sim_data);
	}

} // namespace lcs
