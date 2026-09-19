#pragma once

#include "CollisionDetector/lbvh.h"
#include "Core/scalar.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "SimulationCore/simulation_type.h"
#include <luisa/core/spin_mutex.h>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_set>
#include <string>
#include <string_view>
#include <span>
#include <mutex>
#include <functional>
#include <memory>
#include <type_traits>
#include <luisa/luisa-compute.h>
#include <Utils/async_compiler.h>
#include <Utils/cpu_parallel.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Profiler utils
namespace
{
	constexpr bool	  PRP_profile_enabled = true;
	constexpr bool	  PRP_profile_export_json = PRP_profile_enabled && true;
	const std::string PRP_profile_output_dir = "ResearchLog/Tmp/";
} // namespace

#define PRP_PROFILE_CONCAT_INNER(a, b) a##b
#define PRP_PROFILE_CONCAT(a, b) PRP_PROFILE_CONCAT_INNER(a, b)
#define PRP_PROFILE_SCOPE(tag) \
	[[maybe_unused]] ScopedPRPProfileSelector<PRP_profile_enabled> PRP_PROFILE_CONCAT(_prp_profile_scope_, __LINE__)(tag)
#define PRP_PROFILE_RESET()                \
	do                                     \
	{                                      \
		if constexpr (PRP_profile_enabled) \
			reset_prp_profile_stats();     \
	}                                      \
	while (false)
#define PRP_PROFILE_EXPORT_JSON(name)                                                \
	do                                                                               \
	{                                                                                \
		if constexpr (PRP_profile_enabled && PRP_profile_export_json)                \
			dump_prp_profile_stats_to_json(PRP_profile_output_dir + name + ".json"); \
	}                                                                                \
	while (false)

namespace lcs
{
	void reset_prp_profile_stats();
	void dump_prp_profile_stats_to_json(std::string_view json_path);
} // namespace lcs

namespace lcs
{

	// Intersection detection kernel
	namespace distance
	{
		bool	  LineIntersection(const float3& StartPoint,
			const float3&					EndPoint,
			const float3&					A,
			const float3&					B,
			const float3&					C,
			float3&							OutBary,
			float&							OutTime);
		Var<bool> LineIntersection(const Var<float3>& StartPoint,
			const Var<float3>&						  EndPoint,
			const Var<float3>&						  A,
			const Var<float3>&						  B,
			const Var<float3>&						  C,
			Var<float3>&							  OutBary,
			Var<float>&								  OutTime);
	} // namespace distance

	std::vector<std::vector<uint>> contour_flood_filling_template(const std::vector<std::vector<uint>>& ef_pair_adj_pairs_ext,
		const std::vector<std::vector<uint>>&															ef_pair_adj_pairs_mesh_flip_weight,
		std::vector<uint>&																				ef_pair_mesh_index);

	class ScopedPRPProfile
	{
	private:
		bool		 active = false;
		std::string	 path;
		luisa::Clock clock;

	public:
		explicit ScopedPRPProfile(std::string_view tag);
		~ScopedPRPProfile();
		ScopedPRPProfile(const ScopedPRPProfile&) = delete;
		ScopedPRPProfile& operator=(const ScopedPRPProfile&) = delete;
		ScopedPRPProfile(ScopedPRPProfile&&) = delete;
		ScopedPRPProfile& operator=(ScopedPRPProfile&&) = delete;
	};

	struct ScopedPRPProfileNoop
	{
		explicit ScopedPRPProfileNoop(std::string_view) {}
	};

	template <bool Enabled>
	using ScopedPRPProfileSelector = std::conditional_t<Enabled, ScopedPRPProfile, ScopedPRPProfileNoop>;

	template <typename T>
	class ConcurrentNestedVector
	{
	private:
		std::vector<std::vector<T>>					 _data;
		mutable std::unique_ptr<luisa::spin_mutex[]> _mutexes;
		size_t										 _mutex_count = 0u;

		[[nodiscard]] auto _mutex_view() noexcept
		{
			return std::span(_mutexes.get(), _mutex_count);
		}

		[[nodiscard]] auto _mutex_view() const noexcept
		{
			return std::span(_mutexes.get(), _mutex_count);
		}

		void _check_outer_index(const size_t outer_idx) const
		{
			if (outer_idx >= _data.size())
			{
				LUISA_ERROR("ConcurrentNestedVector index {} out of range {}.", outer_idx, _data.size());
			}
		}

	public:
		ConcurrentNestedVector() = default;
		explicit ConcurrentNestedVector(const size_t outer_size)
		{
			resize(outer_size);
		}
		ConcurrentNestedVector(const ConcurrentNestedVector&) = delete;
		ConcurrentNestedVector& operator=(const ConcurrentNestedVector&) = delete;
		ConcurrentNestedVector(ConcurrentNestedVector&& other) noexcept
			: _data(std::move(other._data))
			, _mutexes(std::move(other._mutexes))
			, _mutex_count(other._mutex_count)
		{
			other._mutex_count = 0u;
		}
		ConcurrentNestedVector& operator=(ConcurrentNestedVector&& other) noexcept
		{
			if (this != &other)
			{
				_data = std::move(other._data);
				_mutexes = std::move(other._mutexes);
				_mutex_count = other._mutex_count;
				other._mutex_count = 0u;
			}
			return *this;
		}

		void resize(const size_t outer_size)
		{
			_data.resize(outer_size);
			_mutexes = outer_size == 0u ? nullptr : std::make_unique<luisa::spin_mutex[]>(outer_size);
			_mutex_count = outer_size;
		}

		[[nodiscard]] size_t size() const noexcept { return _data.size(); }

		[[nodiscard]] std::vector<std::vector<T>>&		 unsafe_data() noexcept { return _data; }
		[[nodiscard]] const std::vector<std::vector<T>>& unsafe_data() const noexcept { return _data; }

		[[nodiscard]] std::vector<T>& unsafe_row(const size_t outer_idx)
		{
			_check_outer_index(outer_idx);
			return _data[outer_idx];
		}

		[[nodiscard]] const std::vector<T>& unsafe_row(const size_t outer_idx) const
		{
			_check_outer_index(outer_idx);
			return _data[outer_idx];
		}

		template <typename U>
		void push_back(const size_t outer_idx, U&& value)
		{
			_check_outer_index(outer_idx);
			std::lock_guard<luisa::spin_mutex> guard(_mutex_view()[outer_idx]);
			_data[outer_idx].push_back(std::forward<U>(value));
		}
	};

	template <typename T>
	T get_weighted_mean(const std::vector<T>& values, const T& zero, const std::vector<float>& weights = {})
	{
		if (values.empty())
			LUISA_ERROR("get_weighted_mean requires at least one value.");
		if (!weights.empty())
		{
			if (values.size() != weights.size())
				LUISA_ERROR("get_weighted_mean value count {} != weight count {}.", values.size(), weights.size());
			T	  weighted_sum = zero;
			T	  unweighted_sum = zero;
			float sum_weights = 0.0f;
			for (size_t i = 0u; i < values.size(); ++i)
			{
				const float weight = weights[i];
				unweighted_sum += values[i];
				weighted_sum += values[i] * weight;
				sum_weights += weight;
			}
			if (sum_weights == 0.0f)
				return unweighted_sum / float(values.size());
			return weighted_sum / sum_weights;
		}
		else
		{
			T sum = zero;
			for (const auto& value : values)
				sum += value;
			return sum / float(values.size());
		}
	}

	namespace RayCasting
	{
		struct HitInfo;
		struct CachedBVH;
		struct BVHCache;
	} // namespace RayCasting

	float  compute_effective_inverse_mass_weight(const RayCasting::HitInfo& hit,
		const std::vector<float>&										   vert_masses);
	float3 compute_inner_lda_direction(const std::vector<float3>& points, const std::vector<float>& weights);

	/// Return all 3 covariance eigenvectors sorted by ascending eigenvalue; eigenvectors[0] is the smallest-eigenvalue direction.
	std::array<float3, 3> compute_all_eigenvectors(const float3x3& S_w);

	namespace SetOperation
	{
		std::vector<uint> get_elements_in_both_A_and_B(const std::vector<uint>& set_a, const std::vector<uint>& set_b);
		std::vector<uint> A_remove_B(const std::vector<uint>& set_A, const std::vector<uint>& set_B);
	} // namespace SetOperation

	namespace RayCasting
	{
		// Hit type enum for distinguishing VF and EE hits
		enum class HitType : uint
		{
			VF = 0, // Vertex-Face hit
			EE = 1, // Edge-Edge hit
			FV = 2,
		};

		struct HitInfo
		{
			HitType hit_type; // TODO: Compress it into src_id/dst_id for 64-bit storage
			uint	group_idx;
			uint	src_mesh_idx;

			uint				src_id; // For VF: vert id; For EE: edge1 id; For FV: face id
			uint				dst_id; // For VF: face id; For EE: edge2 id; For FV: vert id
			std::array<uint, 3> src_verts;
			std::array<uint, 3> dst_verts;

			std::array<float, 2> bary;
			std::array<float, 3> direction;
			float				 dist;
			// float				 score;

			// Helper functions
			bool is_vf() const { return hit_type == HitType::VF; }
			bool is_ee() const { return hit_type == HitType::EE; }
			bool is_fv() const { return hit_type == HitType::FV; }

			uint get_vid() const { return is_vf() ? src_id : dst_id; }
			uint get_fid() const { return is_vf() ? dst_id : src_id; }
			// Legacy direct-ID accessors replaced by hit-type-aware get_vid()/get_fid().
			uint get_eid1() const { return src_id; }
			uint get_eid2() const { return dst_id; }

			uint2 get_edge1() const { return luisa::make_uint2(src_verts[0], src_verts[1]); }
			uint2 get_edge2() const { return luisa::make_uint2(dst_verts[0], dst_verts[1]); }
			uint3 get_face() const { return is_vf() ? uint3{ dst_verts[0], dst_verts[1], dst_verts[2] } : uint3{ src_verts[0], src_verts[1], src_verts[2] }; }

			float3 get_direction() const { return luisa::make_float3(direction[0], direction[1], direction[2]); }
			float3 get_face_bary() const
			{
				return luisa::make_float3(bary[0], bary[1], 1.0f - bary[0] - bary[1]);
			}
			float2 get_edge1_bary() const { return luisa::make_float2(bary[0], 1.0f - bary[0]); }
			float2 get_edge2_bary() const { return luisa::make_float2(bary[1], 1.0f - bary[1]); }

			float4 get_weights() const
			{
				if (is_vf())
				{
					return luisa::make_float4(1.0f, -bary[0], -bary[1], bary[0] + bary[1] - 1.0f);
				}
				else if (is_ee())
				{
					return luisa::make_float4(bary[0], 1.0f - bary[0], -bary[1], bary[1] - 1.0f);
				}
				else if (is_fv())
				{
					return luisa::make_float4(bary[0], bary[1], 1.0f - bary[0] - bary[1], -1.0f);
				}
				else
				{
					return luisa::make_float4(0.0f);
				}
			}

			// Get indices for VF pair: vert, face_vert0, face_vert1, face_vert2
			uint4 get_vf_indices() const
			{
				return luisa::make_uint4(src_verts[0], dst_verts[0], dst_verts[1], dst_verts[2]);
			}

			// Get indices for EE pair: edge1_v0, edge1_v1, edge2_v0, edge2_v1
			uint4 get_ee_indices() const
			{
				return luisa::make_uint4(src_verts[0], src_verts[1], dst_verts[0], dst_verts[1]);
			}

			// Get indices for FV pair: face_vert0, face_vert1, face_vert2, vert
			uint4 get_fv_indices() const
			{
				return luisa::make_uint4(src_verts[0], src_verts[1], src_verts[2], dst_verts[0]);
			}

			// Get all involved vertex ids
			std::vector<uint> get_all_verts() const
			{
				if (is_vf())
					return { src_verts[0], dst_verts[0], dst_verts[1], dst_verts[2] };
				else if (is_ee())
					return { src_verts[0], src_verts[1], dst_verts[0], dst_verts[1] };
				else if (is_fv())
					return { src_verts[0], src_verts[1], src_verts[2], dst_verts[0] };
				else
					return {};
			}

			// Get mesh indices for all involved vertices (for boundary checking)
			std::array<uint, 4> get_all_mesh_indices() const
			{
				if (is_vf())
					return { src_mesh_idx, src_mesh_idx ^ 1, src_mesh_idx ^ 1, src_mesh_idx ^ 1 };
				else if (is_ee())
					return { src_mesh_idx, src_mesh_idx, src_mesh_idx ^ 1, src_mesh_idx ^ 1 };
				else if (is_fv())
					return { src_mesh_idx, src_mesh_idx, src_mesh_idx, src_mesh_idx ^ 1 };
				else
					return { 0, 0, 0, 0 };
			}

			// Get source element vertices (the element that casts the ray/plane)
			std::vector<uint> get_source_verts() const
			{
				if (is_vf())
					return { src_verts[0] };
				else if (is_ee())
					return { src_verts[0], src_verts[1] };
				else if (is_fv())
					return { src_verts[0], src_verts[1], src_verts[2] };
				else
					return {};
			}

			// Get target element vertices (the element being hit)
			std::vector<uint> get_target_verts() const
			{
				if (is_vf())
					return { dst_verts[0], dst_verts[1], dst_verts[2] };
				else if (is_ee())
					return { dst_verts[0], dst_verts[1] };
				else if (is_fv())
					return { dst_verts[0] };
				else
					return {};
			}

			// Create VF hit
			static HitInfo make_vf_hit(
				uint group, uint src_mesh_idx, uint vid, uint fid, const uint3& face_verts, float2 face_bary, float3 dir, float distance)
			{
				HitInfo hit;
				hit.hit_type = HitType::VF;
				hit.group_idx = group;
				hit.src_mesh_idx = src_mesh_idx;
				hit.src_id = vid;
				hit.dst_id = fid;
				hit.src_verts = { vid, -1u, -1u };
				hit.dst_verts = { face_verts.x, face_verts.y, face_verts.z };
				hit.bary = { face_bary.x, face_bary.y };
				hit.direction = { dir.x, dir.y, dir.z };
				hit.dist = distance;
				return hit;
			}

			static HitInfo make_fv_hit(
				uint group, uint src_mesh_idx, uint vid, uint fid, const uint3& face_verts, float2 face_bary, float3 dir, float distance)
			{
				HitInfo hit;
				hit.hit_type = HitType::FV;
				hit.group_idx = group;
				hit.src_mesh_idx = src_mesh_idx;
				hit.src_id = fid;
				hit.dst_id = vid;
				hit.src_verts = { face_verts.x, face_verts.y, face_verts.z };
				hit.dst_verts = { vid, -1u, -1u };
				hit.bary = { face_bary.x, face_bary.y };
				hit.direction = { dir.x, dir.y, dir.z };
				hit.dist = distance;
				return hit;
			}

			// Create EE hit
			static HitInfo make_ee_hit(uint group,
				uint						src_mesh_idx,
				uint						eid1,
				uint						eid2,
				const uint2&				edge1_verts,
				const uint2&				edge2_verts,
				float						edge1_t,
				float						edge2_t,
				float3						dir,
				float						distance)
			{
				HitInfo hit;
				hit.hit_type = HitType::EE;
				hit.group_idx = group;
				hit.src_mesh_idx = src_mesh_idx;
				hit.src_id = eid1;
				hit.dst_id = eid2;
				hit.src_verts = { edge1_verts.x, edge1_verts.y, -1u };
				hit.dst_verts = { edge2_verts.x, edge2_verts.y, -1u };
				hit.bary = { edge1_t, edge2_t };
				hit.direction = { dir.x, dir.y, dir.z };
				hit.dist = distance;
				return hit;
			}
		};

		enum class BatchedVertRayOutputType
		{
			VF,
			FV,
		};

		struct BatchedVertRayRequest
		{
			uint					 group_idx = 0u;
			uint					 source_id = 0u;
			float3					 contour_direction = luisa::make_float3(0.0f);
			BatchedVertRayOutputType output_type = BatchedVertRayOutputType::VF;
			uint2					 target_range = luisa::make_uint2(0u, std::numeric_limits<uint>::max());
			float					 ray_max_dist = 1.0f;
			const std::vector<uint>* target_vertex_hop_dist = nullptr;
			uint					 target_max_hop = std::numeric_limits<uint>::max();
			const std::vector<uint>* target_primitive_ids = nullptr;
		};

		struct BatchedEdgeRayRequest
		{
			uint   group_idx = 0u;
			uint   source_id = 0u;
			float3 direction = luisa::make_float3(0.0f);
			uint2  target_range = luisa::make_uint2(0u, std::numeric_limits<uint>::max());
			float  ray_max_dist = 1.0f;
			// Phase C-1 target projection floor for BVH pruning; -inf disables this optional filter.
			float					 target_proj_min = -std::numeric_limits<float>::infinity();
			const std::vector<uint>* target_vertex_hop_dist = nullptr;
			uint					 target_max_hop = std::numeric_limits<uint>::max();
			const std::vector<uint>* target_primitive_ids = nullptr;
		};

		struct BatchedRaycastStats
		{
			size_t request_count = 0u;
			size_t visited_node_count = 0u;
			size_t tested_leaf_primitive_count = 0u;
			size_t target_filter_reject_count = 0u;
			size_t target_hop_filter_reject_count = 0u;
			size_t produced_hit_count = 0u;
		};

	} // namespace RayCasting

	enum class PairType
	{
		VF,
		EE,
		FV,
	};
	inline auto pair_type_to_string(const PairType& type)
	{
		switch (type)
		{
			case PairType::VF:
				return "VF";
			case PairType::EE:
				return "EE";
			case PairType::FV:
				return "FV";
			default:
				return "Unknown";
		}
	};
	inline PairType get_pair_type(const RayCasting::HitInfo& h)
	{
		return PairType(uint(h.hit_type));
	};
	inline PairType get_pair_type(const uint2& h)
	{
		// 0: VF, 1: EE, 2: FV
		return PairType(h.x >> 30);
	};
	inline Var<uint> get_pair_type(const Var<uint2>& h)
	{
		// 0: VF, 1: EE, 2: FV
		return (h.x >> 30);
	};

	// Optional CSR bucketing partitions candidate hits by primitive and contour; sparse-slot filtering avoids the full primitive×slot allocation.
	std::vector<uint> find_adjacent_hits(const std::vector<RayCasting::HitInfo>& hits_info,
		const uint																 i,
		const MeshData<std::vector>*											 host_mesh_data,
		const std::vector<std::vector<uint>>&									 vert_contains_pairs,
		const std::vector<std::vector<uint>>&									 edge_contains_pairs,
		const std::vector<std::vector<uint>>&									 face_contains_pairs,
		uint																	 key_stride = 1u,
		uint																	 key_offset = 0u,
		const std::vector<uint>*												 hit_slot = nullptr,
		uint																	 query_slot = 0u);

	namespace HitIndices
	{
		constexpr uint extract_mask = 0x3fffffffu; // lower 30 bits for id, upper 2 bits for type

		static inline void assert_type(const uint2& pair, PairType expected)
		{
			if (get_pair_type(pair) != expected)
			{
				LUISA_ERROR("Type assertion failed! Expected {}, got {}", pair_type_to_string(expected), pair_type_to_string(get_pair_type(pair)));
			}
		};
		static inline void assert_type(const Var<uint2>& pair, PairType expected)
		{
			uint expected_var = uint(expected);
			$if(get_pair_type(pair) != expected_var)
			{
				luisa::compute::device_assert(false, "Type assertion failed");
			};
		};

		inline auto get_vf_vid(const auto& pair)
		{
			assert_type(pair, PairType::VF);
			return pair.x & extract_mask;
		};
		inline auto get_vf_fid(const auto& pair)
		{
			assert_type(pair, PairType::VF);
			return pair.y & extract_mask;
		};
		inline auto get_ee_eid1(const auto& pair)
		{
			assert_type(pair, PairType::EE);
			return pair.x & extract_mask;
		};
		inline auto get_ee_eid2(const auto& pair)
		{
			assert_type(pair, PairType::EE);
			return pair.y & extract_mask;
		};
		inline auto get_fv_fid(const auto& pair)
		{
			assert_type(pair, PairType::FV);
			return pair.x & extract_mask;
		};
		inline auto get_fv_vid(const auto& pair)
		{
			assert_type(pair, PairType::FV);
			return pair.y & extract_mask;
		};
	} // namespace HitIndices

} // namespace lcs
