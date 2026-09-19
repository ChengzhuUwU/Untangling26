#include "intersection_resolver_helper.h"
#include "CollisionDetector/distance.hpp"
#include "Core/lc_to_eigen.h"
#include "Utils/cpu_parallel.h"
#include "luisa/core/mathematics.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <string>

namespace lcs
{
	namespace
	{
		struct PRPProfileEntry
		{
			double total_ms = 0.0;
			uint64 count = 0ull;
		};

		auto& prp_profile_entries()
		{
			static std::unordered_map<std::string, PRPProfileEntry> entries;
			return entries;
		}
		auto& prp_profile_mutex()
		{
			static std::mutex mtx;
			return mtx;
		}
		auto& prp_profile_tag_stack()
		{
			static thread_local std::vector<std::string> tags;
			return tags;
		}
		struct PRPProfileRow
		{
			std::string path;
			uint		depth = 0u;
			std::string parent;
			double		total_ms = 0.0;
			uint64		count = 0ull;
			double		avg_ms = 0.0;
			double		ratio_to_root = 0.0;
		};
		auto collect_prp_profile_rows()
		{
			std::vector<PRPProfileRow> rows;
			{
				std::lock_guard lock(prp_profile_mutex());
				rows.reserve(prp_profile_entries().size());
				for (const auto& [path, entry] : prp_profile_entries())
				{
					const uint		  depth = static_cast<uint>(std::count(path.begin(), path.end(), '/'));
					const auto		  pos = path.rfind('/');
					const std::string parent = (pos == std::string::npos) ? std::string{} : path.substr(0, pos);
					const double	  avg = entry.count == 0u ? 0.0 : entry.total_ms / static_cast<double>(entry.count);
					rows.push_back(PRPProfileRow{
						.path = path,
						.depth = depth,
						.parent = parent,
						.total_ms = entry.total_ms,
						.count = entry.count,
						.avg_ms = avg,
						.ratio_to_root = 0.0 });
				}
			}
			double root_ms = 0.0;
			for (const auto& row : rows)
			{
				if (row.path == "host_resolve_intersections_PRP/total" || row.path == "host_resolve_intersections_PRP.total")
				{
					root_ms = row.total_ms;
					break;
				}
			}
			if (root_ms <= 1e-12)
			{
				for (const auto& row : rows)
					root_ms = std::max(root_ms, row.total_ms);
			}
			if (root_ms > 1e-12)
			{
				for (auto& row : rows)
					row.ratio_to_root = row.total_ms / root_ms;
			}
			return rows;
		}
		void ensure_parent_dir_exists(std::string_view path)
		{
			const std::filesystem::path p(path);
			if (!p.parent_path().empty())
				std::filesystem::create_directories(p.parent_path());
		}
		std::string escape_json_string(const std::string& s)
		{
			std::string out;
			out.reserve(s.size() + 8u);
			for (const char c : s)
			{
				switch (c)
				{
					case '\\':
						out += "\\\\";
						break;
					case '"':
						out += "\\\"";
						break;
					case '\n':
						out += "\\n";
						break;
					case '\r':
						out += "\\r";
						break;
					case '\t':
						out += "\\t";
						break;
					default:
						out.push_back(c);
						break;
				}
			}
			return out;
		}

		std::string normalize_profile_tag(std::string_view tag)
		{
			std::string normalized;
			normalized.reserve(tag.size() + 4u);
			bool prev_is_sep = true;
			for (const char c : tag)
			{
				const bool is_sep = (c == '.') || (c == '/');
				if (is_sep)
				{
					if (!prev_is_sep)
						normalized.push_back('/');
					prev_is_sep = true;
				}
				else
				{
					normalized.push_back(c);
					prev_is_sep = false;
				}
			}
			while (!normalized.empty() && normalized.back() == '/')
				normalized.pop_back();
			return normalized;
		}

		std::vector<std::string_view> split_path_segments(std::string_view path)
		{
			std::vector<std::string_view> segs;
			size_t						  start = 0u;
			while (start < path.size())
			{
				const size_t end = path.find('/', start);
				if (end == std::string_view::npos)
				{
					if (start < path.size())
						segs.emplace_back(path.substr(start));
					break;
				}
				if (end > start)
					segs.emplace_back(path.substr(start, end - start));
				start = end + 1u;
			}
			return segs;
		}

		std::string join_path_segments(const std::vector<std::string_view>& segs, size_t begin_idx)
		{
			std::string out;
			for (size_t i = begin_idx; i < segs.size(); i++)
			{
				if (!out.empty())
					out.push_back('/');
				out.append(segs[i]);
			}
			return out;
		}

		std::string relativize_tag_against_parent(const std::string& normalized_tag, const std::string& parent_tag)
		{
			if (normalized_tag.empty() || parent_tag.empty())
				return normalized_tag;
			const auto	 child_segs = split_path_segments(normalized_tag);
			const auto	 parent_segs = split_path_segments(parent_tag);
			const size_t n = std::min(child_segs.size(), parent_segs.size());
			size_t		 common = 0u;
			while (common < n && child_segs[common] == parent_segs[common])
				common++;
			if (common == 0u)
				return normalized_tag;
			if (common >= child_segs.size())
				return std::string(child_segs.back());
			return join_path_segments(child_segs, common);
		}

		std::string prp_profile_push_tag(std::string_view tag)
		{
			auto&		stack = prp_profile_tag_stack();
			std::string normalized = normalize_profile_tag(tag);
			if (!stack.empty())
				normalized = relativize_tag_against_parent(normalized, stack.back());
			if (normalized.empty())
				normalized = "scope";
			stack.emplace_back(std::move(normalized));
			std::string path;
			for (size_t i = 0; i < stack.size(); i++)
			{
				if (i != 0u)
					path.push_back('/');
				path += stack[i];
			}
			return path;
		}

		void prp_profile_pop_tag()
		{
			auto& stack = prp_profile_tag_stack();
			if (!stack.empty())
				stack.pop_back();
		}
	} // namespace

} // namespace lcs

namespace lcs
{
	void reset_prp_profile_stats()
	{
		std::lock_guard lock(prp_profile_mutex());
		prp_profile_entries().clear();
	}

	void dump_prp_profile_stats_to_json(std::string_view json_path)
	{
		try
		{
			auto rows = collect_prp_profile_rows();
			std::sort(rows.begin(), rows.end(),
				[](const PRPProfileRow& lhs, const PRPProfileRow& rhs)
				{
					return lhs.total_ms > rhs.total_ms;
				});
			double root_ms = 0.0;
			for (const auto& row : rows)
				root_ms = std::max(root_ms, row.total_ms);
			ensure_parent_dir_exists(json_path);
			std::ofstream ofs(std::string(json_path), std::ios::out | std::ios::trunc);
			if (!ofs.is_open())
			{
				LUISA_WARNING("Failed to open PRP profile JSON file: {}", json_path);
				return;
			}
			ofs << "{\n";
			ofs << "  \"root_total_ms\": " << root_ms << ",\n";
			ofs << "  \"entry_count\": " << rows.size() << ",\n";
			ofs << "  \"entries\": [\n";
			for (size_t i = 0; i < rows.size(); i++)
			{
				const auto& r = rows[i];
				ofs << "    {\n";
				ofs << "      \"rank\": " << (i + 1u) << ",\n";
				ofs << "      \"path\": \"" << escape_json_string(r.path) << "\",\n";
				ofs << "      \"depth\": " << r.depth << ",\n";
				ofs << "      \"parent\": \"" << escape_json_string(r.parent) << "\",\n";
				ofs << "      \"total_ms\": " << r.total_ms << ",\n";
				ofs << "      \"count\": " << r.count << ",\n";
				ofs << "      \"avg_ms\": " << r.avg_ms << ",\n";
				ofs << "      \"ratio_to_root\": " << r.ratio_to_root << "\n";
				ofs << "    }" << (i + 1u == rows.size() ? "\n" : ",\n");
			}
			ofs << "  ]\n";
			ofs << "}\n";
			ofs.close();
			LUISA_INFO("[PRP] Profile JSON exported: {}", json_path);
		}
		catch (const std::exception& e)
		{
			LUISA_WARNING("Failed to export PRP profile JSON ({}): {}", json_path, e.what());
		}
	}

	ScopedPRPProfile::ScopedPRPProfile(std::string_view tag)
	{
		path = prp_profile_push_tag(tag);
		active = true;
	}

	ScopedPRPProfile::~ScopedPRPProfile()
	{
		if (active)
		{
			const double elapsed_ms = clock.toc();
			{
				std::lock_guard lock(prp_profile_mutex());
				auto&			entry = prp_profile_entries()[path];
				entry.total_ms += elapsed_ms;
				entry.count++;
			}
			prp_profile_pop_tag();
		}
	}

} // namespace lcs

namespace lcs
{
	namespace distance
	{
		namespace detail
		{
			constexpr float UE_SMALL_NUMBER = 1e-6f;
			constexpr float UE_SMALL_NUMBER_SQUARED = UE_SMALL_NUMBER * UE_SMALL_NUMBER;
			// template <typename T>
			struct TRayTriangleIntersectionDefaultToleranceProvider
			{
				static constexpr float ParallelTolerance = (float)UE_SMALL_NUMBER;
				static constexpr float BaryTolerance = (float)UE_SMALL_NUMBER;
			};

			template <typename T, typename ToleranceProvider = TRayTriangleIntersectionDefaultToleranceProvider>
			inline bool RayTriangleIntersectionAndBary(const luisa::float3& RayStart,
				const luisa::float3&										RayDir,
				T															RayLength,
				const luisa::float3&										A,
				const luisa::float3&										B,
				const luisa::float3&										C,
				T&															OutT,
				luisa::float3&												OutBary,
				luisa::float3&												OutN)
			{
				const luisa::float3 AB = B - A; // edge 1
				const luisa::float3 AC = C - A; // edge 2
				const luisa::float3 Normal = luisa::cross(AB, AC);
				const luisa::float3 NegRayDir = -RayDir;

				const T NormalLength = luisa::length(Normal);
				const T Den = luisa::dot(NegRayDir, Normal);
				if (luisa::abs(Den) < NormalLength * ToleranceProvider::ParallelTolerance)
				{
					// ray is parallel or away to the triangle plane it is a miss
					return false;
				}

				const T InvDen = (T)1 / Den;

				// let's compute the time to intersection
				const luisa::float3 RayToA = RayStart - A;
				const T				Time = luisa::dot(RayToA, Normal) * InvDen;
				if (Time < 0.0f || Time > RayLength)
				{
					return false;
				}

				// now compute barycentric coordinates
				const luisa::float3 RayToACrossNegDir = luisa::cross(NegRayDir, RayToA);
				const T				UU = luisa::dot(AC, RayToACrossNegDir) * InvDen;
				if (UU < ToleranceProvider::BaryTolerance || UU > (1 - ToleranceProvider::BaryTolerance))
				{
					return false; // outside of the triangle
				}
				const T VV = -luisa::dot(AB, RayToACrossNegDir) * InvDen;
				if (VV < ToleranceProvider::BaryTolerance || (VV + UU) > (1 - ToleranceProvider::BaryTolerance))
				{
					return false; // outside of the triangle
				}

				// point is within the triangle, let's compute
				OutT = Time;
				OutBary = { 1.0f - UU - VV, UU, VV };
				OutN = luisa::normalize(Normal);
				OutN *= luisa::sign(Den);
				return true;
			}

			template <typename T, typename ToleranceProvider = TRayTriangleIntersectionDefaultToleranceProvider>
			inline Var<bool> RayTriangleIntersectionAndBary(const Var<luisa::float3>& RayStart,
				const Var<luisa::float3>&											  RayDir,
				T																	  RayLength,
				const Var<luisa::float3>&											  A,
				const Var<luisa::float3>&											  B,
				const Var<luisa::float3>&											  C,
				T&																	  OutT,
				Var<luisa::float3>&													  OutBary,
				Var<luisa::float3>&													  OutN)
			{
				const Var<luisa::float3> AB = B - A; // edge 1
				const Var<luisa::float3> AC = C - A; // edge 2
				const Var<luisa::float3> Normal = luisa::compute::cross(AB, AC);
				const Var<luisa::float3> NegRayDir = -RayDir;

				const T	  NormalLength = luisa::compute::length(Normal);
				const T	  Den = luisa::compute::dot(NegRayDir, Normal);
				Var<bool> result = false;
				$if(luisa::compute::abs(Den) < NormalLength * ToleranceProvider::ParallelTolerance)
				{
					// nearly parallel
				}
				$else
				{
					const T InvDen = (T)1 / Den;

					// let's compute the time to intersection
					const Var<luisa::float3> RayToA = RayStart - A;
					const T					 Time = luisa::compute::dot(RayToA, Normal) * InvDen;
					$if(Time<0.0f | Time> RayLength)
					{
						//
					}
					$else
					{
						// now compute barycentric coordinates
						const Var<luisa::float3> RayToACrossNegDir = luisa::compute::cross(NegRayDir, RayToA);
						const T					 UU = luisa::compute::dot(AC, RayToACrossNegDir) * InvDen;
						$if(UU<ToleranceProvider::BaryTolerance | UU>(1.0f - ToleranceProvider::BaryTolerance))
						{
							// outside of the triangle
						}
						$else
						{
							const T VV = -luisa::compute::dot(AB, RayToACrossNegDir) * InvDen;
							$if(VV<ToleranceProvider::BaryTolerance | (VV + UU)>(1.0f - ToleranceProvider::BaryTolerance))
							{
								// outside of the triangle
							}
							$else
							{
								// point is within the triangle, let's compute
								OutT = Time;
								OutBary = { 1.0f - UU - VV, UU, VV };
								OutN = luisa::compute::normalize(Normal);
								OutN *= luisa::compute::sign(Den);
								result = true;
							};
						};
					};
				};
				return result;
			}
		} // namespace detail

		// template <typename T>
		bool LineIntersection(const float3& StartPoint,
			const float3&					EndPoint,
			const float3&					A,
			const float3&					B,
			const float3&					C,
			float3&							OutBary,
			float&							OutTime)
		{
			const float3 StartToEnd = EndPoint - StartPoint;
			const float	 SegmentLenSq = length_squared_vec(StartToEnd);
			if (SegmentLenSq < detail::UE_SMALL_NUMBER_SQUARED)
			{
				return false;
			}
			const float	 SegmentLen = luisa::sqrt(SegmentLenSq);
			const float	 OneOverSegmentLen = (float)1 / SegmentLen;
			const float3 Ray = StartToEnd * OneOverSegmentLen;

			float3 NormalUnused;
			if (detail::RayTriangleIntersectionAndBary(StartPoint, Ray, SegmentLen, A, B, C, OutTime, OutBary, NormalUnused))
			{
				// OutTime is between 0 and SegmentLen. Convert to 0 to 1
				OutTime *= OneOverSegmentLen;
				return true;
			}

			return false;
		}
		// template <typename T>
		Var<bool> LineIntersection(const Var<float3>& StartPoint,
			const Var<float3>&						  EndPoint,
			const Var<float3>&						  A,
			const Var<float3>&						  B,
			const Var<float3>&						  C,
			Var<float3>&							  OutBary,
			Var<float>&								  OutTime)
		{
			const Var<float3> StartToEnd = EndPoint - StartPoint;
			const Var<float>  SegmentLenSq = length_squared_vec(StartToEnd);

			Var<bool> result = false;
			$if(SegmentLenSq < detail::UE_SMALL_NUMBER_SQUARED)
			{
			}
			$else
			{
				const Var<float>  SegmentLen = luisa::compute::sqrt(SegmentLenSq);
				const Var<float>  OneOverSegmentLen = 1.0f / SegmentLen;
				const Var<float3> Ray = StartToEnd * OneOverSegmentLen;

				Var<float3> NormalUnused;
				$if(detail::RayTriangleIntersectionAndBary(StartPoint, Ray, SegmentLen, A, B, C, OutTime, OutBary, NormalUnused))
				{
					// OutTime is between 0 and SegmentLen. Convert to 0 to 1
					OutTime *= OneOverSegmentLen;
					result = true;
				}
				$else{};
			};
			return result;
		}

	}; // namespace distance

	std::vector<std::vector<uint>> contour_flood_filling_template(const std::vector<std::vector<uint>>& ef_pair_adj_pairs_ext,
		const std::vector<std::vector<uint>>&															ef_pair_adj_pairs_mesh_flip_weight,
		std::vector<uint>&																				ef_pair_mesh_index)
	{
		const uint						num_pairs = ef_pair_adj_pairs_ext.size();
		std::vector<std::vector<uint2>> groups;
		std::vector<bool>				visited(num_pairs, false);
		ef_pair_mesh_index.resize(num_pairs, -1u);

		for (uint index = 0; index < num_pairs; index++)
		{
			if (visited[index])
				continue;

			std::vector<uint2> curr_group = { luisa::make_uint2(index, 0) };
			std::vector<uint2> to_visit = { luisa::make_uint2(index, 0) };
			visited[index] = true;

			while (!to_visit.empty())
			{
				const uint2 pair = to_visit.back();
				const uint	pair_idx = pair.x;
				const uint	pair_sum = pair.y;
				to_visit.pop_back();

				const auto& adj_pairs = ef_pair_adj_pairs_ext[pair_idx];
				const auto& adj_pairs_weight = ef_pair_adj_pairs_mesh_flip_weight[pair_idx];
				for (uint j = 0; j < adj_pairs.size(); j++)
				{
					const uint adj_pair_idx = adj_pairs[j];
					const uint adj_pair_info = adj_pairs_weight[j];
					if (!visited[adj_pair_idx])
					{
						uint2 adj_pair = luisa::make_uint2(adj_pair_idx, pair_sum);
						adj_pair.y += adj_pair_info; // same_mesh
						visited[adj_pair_idx] = true;
						to_visit.push_back(adj_pair);
						curr_group.emplace_back(adj_pair);
					}
				}
			}
			groups.emplace_back(curr_group);
		}
		std::vector<std::vector<uint>> contour;
		for (const auto& group : groups)
		{
			std::vector<uint> contour_indices;
			for (const auto& pair : group)
			{
				contour_indices.push_back(pair.x);
				const uint mesh_idx_sum = pair.y;
				const uint mesh_idx = mesh_idx_sum % 2;
				ef_pair_mesh_index[pair.x] = mesh_idx;
			}
			contour.push_back(contour_indices);
		}
		return contour;
	};
	static bool list_contains(const std::vector<uint>& list, const uint val)
	{
		return std::find(list.begin(), list.end(), val) != list.end();
	}
	inline bool is_finite_matrix(const float3x3& m)
	{
		for (uint c = 0; c < 3; c++)
			if (!std::isfinite(m[c].x) || !std::isfinite(m[c].y) || !std::isfinite(m[c].z))
				return false;
		return true;
	}

	bool eigen_decomposition_symmetric(const EigenFloat3x3& S, EigenFloat3x3& eigenvectors, EigenFloat3& eigenvalues)
	{
		if (!S.allFinite())
		{
			eigenvectors = EigenFloat3x3::Identity();
			eigenvalues.setZero();
			return false;
		}
		Eigen::SelfAdjointEigenSolver<EigenFloat3x3> solver;
		solver.computeDirect(S); // analytic 3x3 path, more stable than QR iterations
		if (solver.info() != Eigen::Success)
		{
			LUISA_WARNING("Eigen decomposition failed (status={}), falling back to identity.", static_cast<int>(solver.info()));
			std::cout << "Input matrix S:\n"
					  << S << std::endl;
			eigenvectors = EigenFloat3x3::Identity();
			eigenvalues.setZero();
			return false;
		}
		eigenvectors = solver.eigenvectors();
		eigenvalues = solver.eigenvalues();
		return true;
	}
	float compute_effective_inverse_mass_weight(const RayCasting::HitInfo& hit, const std::vector<float>& vert_masses)
	{
		auto safe_inverse_mass = [&](const uint vid)
		{
			const float mass = std::max(vert_masses[vid], 1e-8f);
			return 1.0f / mass;
		};

		if (hit.is_vf())
		{
			const float3 bary = hit.get_face_bary();
			return safe_inverse_mass(hit.src_verts[0])
				+ bary.x * bary.x * safe_inverse_mass(hit.dst_verts[0])
				+ bary.y * bary.y * safe_inverse_mass(hit.dst_verts[1])
				+ bary.z * bary.z * safe_inverse_mass(hit.dst_verts[2]);
		}
		if (hit.is_fv())
		{
			const float3 bary = hit.get_face_bary();
			return bary.x * bary.x * safe_inverse_mass(hit.src_verts[0])
				+ bary.y * bary.y * safe_inverse_mass(hit.src_verts[1])
				+ bary.z * bary.z * safe_inverse_mass(hit.src_verts[2])
				+ safe_inverse_mass(hit.dst_verts[0]);
		}
		if (hit.is_ee())
		{
			const float2 bary1 = hit.get_edge1_bary();
			const float2 bary2 = hit.get_edge2_bary();
			return bary1.x * bary1.x * safe_inverse_mass(hit.src_verts[0])
				+ bary1.y * bary1.y * safe_inverse_mass(hit.src_verts[1])
				+ bary2.x * bary2.x * safe_inverse_mass(hit.dst_verts[0])
				+ bary2.y * bary2.y * safe_inverse_mass(hit.dst_verts[1]);
		}
		return 1.0f;
	}

	float3 compute_inner_lda_direction(const std::vector<float3>& points, const std::vector<float>& weights)
	{
		const uint num = points.size();
		if (num == 0)
			LUISA_ERROR("Cannot compute LDA direction with zero points.");
		float3 mean = get_weighted_mean(points, luisa::make_float3(0.0f), weights);
		// Within-class scatter
		float3x3 cov = luisa::make_float3x3(0.0f);
		for (uint i = 0; i < num; i++)
		{
			const float3 diff = points[i] - mean;
			cov = cov + weights[i] * outer_product(diff, diff);
		}
		float3x3 S_w = cov;

		if (!is_finite_matrix(S_w))
			return luisa::make_float3(0.0f, 1.0f, 0.0f);
		const float scatter_trace = S_w[0].x + S_w[1].y + S_w[2].z;
		if (scatter_trace < 1e-10f)
			return luisa::make_float3(0.0f, 1.0f, 0.0f);

		EigenFloat3x3 _S = float3x3_to_eigen3x3(S_w);
		EigenFloat3x3 _eigenvectors;
		EigenFloat3	  _eigenvalues;
		if (!eigen_decomposition_symmetric(_S, _eigenvectors, _eigenvalues))
			return luisa::make_float3(0.0f, 1.0f, 0.0f);
		float3 lda_dir = eigen3_to_float3(_eigenvectors.col(0)); // Minimum eigenvalue direction
		if (length_squared_vec(lda_dir) < 1e-12f)
			return luisa::make_float3(0.0f, 1.0f, 0.0f);

		return luisa::normalize(lda_dir);
	}
	std::array<float3, 3> compute_all_eigenvectors(const float3x3& S_w)
	{
		const std::array<float3, 3> default_basis = {
			luisa::make_float3(0, 1, 0), luisa::make_float3(1, 0, 0), luisa::make_float3(0, 0, 1)
		};

		if (!is_finite_matrix(S_w))
			return default_basis;

		const float scatter_trace = S_w[0].x + S_w[1].y + S_w[2].z;
		if (scatter_trace < 1e-10f)
			return default_basis;

		// Guard against near-zero covariance using the Frobenius norm, not only the trace.
		float frob_sq = 0.0f;
		for (uint c = 0; c < 3; c++)
			frob_sq += luisa::dot(S_w[c], S_w[c]);
		if (frob_sq < 1e-20f)
			return default_basis;

		EigenFloat3x3 _S = float3x3_to_eigen3x3(S_w);
		EigenFloat3x3 _eigenvectors;
		EigenFloat3	  _eigenvalues;
		if (!eigen_decomposition_symmetric(_S, _eigenvectors, _eigenvalues))
			return default_basis;
		// Eigen::SelfAdjointEigenSolver returns eigenvalues in ascending order. col(0) = smallest, col(1) = 2nd, col(2) = largest.
		auto safe_normalize = [&](const float3& v, const float3& fallback) -> float3
		{
			return length_squared_vec(v) > 1e-12f ? luisa::normalize(v) : fallback;
		};
		return {
			safe_normalize(eigen3_to_float3(_eigenvectors.col(0)), default_basis[0]),
			safe_normalize(eigen3_to_float3(_eigenvectors.col(1)), default_basis[1]),
			safe_normalize(eigen3_to_float3(_eigenvectors.col(2)), default_basis[2]),
		};
	}

	namespace SetOperation
	{
		std::vector<uint> get_elements_in_both_A_and_B(const std::vector<uint>& set_a, const std::vector<uint>& set_b)
		{
			std::vector<uint> intersection;
			std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(), std::back_inserter(intersection));
			return intersection;
		}
		std::vector<uint> A_remove_B(const std::vector<uint>& set_A, const std::vector<uint>& set_B)
		{
			std::vector<uint> culled_set;
			std::set_difference(set_A.begin(), set_A.end(), set_B.begin(), set_B.end(), std::back_inserter(culled_set));
			return culled_set;
		}
	} // namespace SetOperation

	bool edge_contains_vert(const MeshData<std::vector>* host_mesh_data, const uint eid, const uint vid)
	{
		uint2 edge_verts = host_mesh_data->sa_edges[eid];
		return luisa::any(edge_verts == vid);
	};
	auto face_contains_vert(const MeshData<std::vector>* host_mesh_data, const uint fid, const uint vid)
	{
		uint3 face_verts = host_mesh_data->sa_faces[fid];
		return luisa::any(face_verts == vid);
	};
	auto face_contains_edge(const MeshData<std::vector>* host_mesh_data, const uint fid, const uint eid)
	{
		return list_contains(host_mesh_data->edge_adj_faces_ext[eid], fid);
	};
	auto edges_adjacent(const MeshData<std::vector>* host_mesh_data, const uint eid1, const uint eid2)
	{
		return list_contains(host_mesh_data->edge_adj_edges_ext[eid1], eid2);
	};
	auto verts_adjacent(const MeshData<std::vector>* host_mesh_data, const uint vid1, const uint vid2)
	{
		return list_contains(host_mesh_data->vert_adj_verts[vid1], vid2);
	};

	template <uint N>
	std::vector<uint> vec_to_list(const luisa::Vector<uint, N>& arr);

	std::vector<uint> vec_to_list(const luisa::Vector<uint, 2>& arr)
	{
		return { arr.x, arr.y };
	};
	std::vector<uint> vec_to_list(const luisa::Vector<uint, 3>& arr)
	{
		return { arr.x, arr.y, arr.z };
	};
	std::vector<uint> vec_to_list(const luisa::Vector<uint, 4>& arr)
	{
		return { arr.x, arr.y, arr.z, arr.w };
	}

	std::vector<uint> find_adjacent_hits(const std::vector<RayCasting::HitInfo>& hits_info,
		const uint																 i,
		const MeshData<std::vector>*											 host_mesh_data,
		const std::vector<std::vector<uint>>&									 vert_contains_pairs,
		const std::vector<std::vector<uint>>&									 edge_contains_pairs,
		const std::vector<std::vector<uint>>&									 face_contains_pairs,
		uint																	 key_stride,
		uint																	 key_offset,
		const std::vector<uint>*												 hit_slot,
		uint																	 query_slot)
	{
		const auto& left = hits_info[i];
		PairType	L = get_pair_type(left);

		std::vector<uint> adj_pairs;
		adj_pairs.reserve(16u);

		// CSR-bucket key remap. `key_stride == 1, key_offset == 0` → legacy direct lookup.
		auto kv = [key_stride, key_offset](uint x)
		{ return x * key_stride + key_offset; };

		auto try_pair_v2 = [&](uint j)
		{
			if (i == j)
				return;
			if (hit_slot != nullptr && (*hit_slot)[j] != query_slot)
				return;
			adj_pairs.push_back(j);
		};

		// ========================= VF =========================
		if (L == PairType::VF)
		{
			uint vid = left.get_vid();
			uint fid = left.get_fid();

			// VF adj VF: Access left.V adj verts, Query if verts.VFpair.F == left.F
			for (uint av : host_mesh_data->vert_adj_verts[vid])
			{
				for (uint j : vert_contains_pairs[kv(av)])
					if (get_pair_type(hits_info[j]) == PairType::VF
						&& left.get_fid() == hits_info[j].get_fid())
						try_pair_v2(j);
			}

			// VF adj FV: Access left.V adj faces, Query if left.F contains verts.FVpair.V
			for (uint af : host_mesh_data->vert_adj_faces[vid])
				for (uint j : face_contains_pairs[kv(af)])
					if (get_pair_type(hits_info[j]) == PairType::FV
						&& face_contains_vert(host_mesh_data, fid, hits_info[j].get_vid()))
						try_pair_v2(j);

			// VF adj EE: Access left.V adj edges, Query if left.F contains edges.EEpair.E2
			for (uint ae : host_mesh_data->vert_adj_edges[vid])
				for (uint j : edge_contains_pairs[kv(ae)])
					if (get_pair_type(hits_info[j]) == PairType::EE
						&& face_contains_edge(host_mesh_data, fid, hits_info[j].get_eid2()))
						try_pair_v2(j);
		}

		// ========================= FV =========================
		else if (L == PairType::FV)
		{
			uint vid = left.get_vid();
			uint fid = left.get_fid();

			// FV adj VF: Access left.F adj verts, Query if verts.VFpair.F contains left.V
			for (uint av : vec_to_list(host_mesh_data->sa_faces[fid]))
			{
				for (uint j : vert_contains_pairs[kv(av)])
					if (get_pair_type(hits_info[j]) == PairType::VF
						&& face_contains_vert(host_mesh_data, hits_info[j].get_fid(), vid))
						try_pair_v2(j);
			}

			// FV adj FV: Access left.F, Query if F.FVpair.V adj left.V
			for (uint j : face_contains_pairs[kv(fid)])
				if (get_pair_type(hits_info[j]) == PairType::FV
					&& verts_adjacent(host_mesh_data, vid, hits_info[j].get_vid()))
				{
					try_pair_v2(j);
				}

			// FV adj EE: Access left.F adj edges, Query if edges.EEpair.E2 contains left.V
			for (uint ae : vec_to_list(host_mesh_data->face_adj_edges[fid]))
				for (uint j : edge_contains_pairs[kv(ae)])
					if (get_pair_type(hits_info[j]) == PairType::EE
						&& edge_contains_vert(host_mesh_data, hits_info[j].get_eid2(), vid))
						try_pair_v2(j);
		}

		// ========================= EE =========================
		else if (L == PairType::EE)
		{
			uint eid1 = left.get_eid1();
			uint eid2 = left.get_eid2();

			// EE adj VF: Access left.E1 adj verts, Query if verts.VFpair.F contains left.E2
			for (uint av : vec_to_list(host_mesh_data->sa_edges[eid1]))
			{
				for (uint j : vert_contains_pairs[kv(av)])
					if (get_pair_type(hits_info[j]) == PairType::VF
						&& face_contains_edge(host_mesh_data, hits_info[j].get_fid(), eid2))
						try_pair_v2(j);
			}

			// EE adj FV: Access left.E1 adj faces, Query if left.E2 contains faces.FVpair.V
			for (uint af : host_mesh_data->edge_adj_faces_ext[eid1])
				for (uint j : face_contains_pairs[kv(af)])
					if (get_pair_type(hits_info[j]) == PairType::FV
						&& edge_contains_vert(host_mesh_data, eid2, hits_info[j].get_vid()))
						try_pair_v2(j);

			// EE adj EE:    left.E1 == right.E1, right.E2 adj left.E2 or left.E2 == right.E2, right.E1 adj left.E1

			// EE adj EE: Access left.E1, Query if edges.EEpair.E2 adj left.E2
			for (uint j : edge_contains_pairs[kv(eid1)]) // TODO???
				if (get_pair_type(hits_info[j]) == PairType::EE && hits_info[j].get_eid1() == eid1
					&& edges_adjacent(host_mesh_data, eid2, hits_info[j].get_eid2()))
					try_pair_v2(j);

			// EE adj EE: Access left.E1 adj edges, Query if edges.EEpair.E2 == left.E2
			for (uint ae : host_mesh_data->edge_adj_edges_ext[eid1])
				for (uint j : edge_contains_pairs[kv(ae)]) // TODO???
					if (get_pair_type(hits_info[j]) == PairType::EE && hits_info[j].get_eid2() == eid2
						// && edges_adjacent(host_mesh_data, eid1, hits_info[j].get_eid1())
					)
						try_pair_v2(j);
		}

		return adj_pairs;
	}

} // namespace lcs
