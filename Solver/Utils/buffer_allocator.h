#pragma once

#include "luisa/runtime/device.h"
#include <vector>

namespace lcs::Initializer
{

	template <typename T>
	static inline auto upload_buffer(luisa::compute::Device& device,
		luisa::compute::Buffer<T>&							 dest,
		const std::vector<T>&								 src)
	{
		if (!dest.valid())
		{
			dest = device.create_buffer<T>(src.size());
		}
		return dest.copy_from(src.data());
	};

	template <typename T>
	inline void upload_from(std::vector<T>& dest, const std::vector<T>& input_data)
	{
		dest.resize(input_data.size());
		std::memcpy(dest.data(), input_data.data(), dest.size() * sizeof(T));
	}
	inline uint upload_2d_csr_from(std::vector<uint>& dest, const std::vector<std::vector<uint>>& input_map)
	{
		uint num_outer = input_map.size();
		uint current_prefix = num_outer + 1;

		std::vector<uint> prefix_list(num_outer + 1);

		uint max_count = 0;
		for (uint i = 0; i < num_outer; i++)
		{
			const auto& inner_list = input_map[i];
			uint		num_inner = inner_list.size();
			max_count = std::max(max_count, num_inner);
			prefix_list[i] = current_prefix;
			current_prefix += num_inner;
		}
		uint num_data = current_prefix;
		prefix_list[num_outer] = current_prefix;

		dest.resize(num_data);
		std::memcpy(dest.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

		for (uint i = 0; i < num_outer; i++)
		{
			const auto& inner_list = input_map[i];
			uint		current_prefix = prefix_list[i];
			uint		current_end = prefix_list[i + 1];
			for (uint j = current_prefix; j < current_end; j++)
			{
				dest[j] = inner_list[j - current_prefix];
			}
		}
		return max_count;
	}
	inline uint upload_2d_csr_from(std::vector<uint>& dest_prefix, std::vector<uint>& dest_value, const std::vector<std::vector<uint>>& input_map)
	{
		uint num_outer = input_map.size();
		dest_prefix.resize(num_outer + 1);

		std::vector<uint> prefix_list(num_outer + 1);

		uint current_prefix = 0;
		uint max_count = 0;
		for (uint i = 0; i < num_outer; i++)
		{
			const auto& inner_list = input_map[i];
			uint		num_inner = inner_list.size();
			max_count = std::max(max_count, num_inner);
			prefix_list[i] = current_prefix;
			current_prefix += num_inner;
		}
		prefix_list[num_outer] = current_prefix;

		uint num_data = current_prefix;
		dest_value.resize(num_data);
		std::memcpy(dest_prefix.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

		for (uint i = 0; i < num_outer; i++)
		{
			const auto& inner_list = input_map[i];
			uint		current_prefix = prefix_list[i];
			uint		current_end = prefix_list[i + 1];
			for (uint j = current_prefix; j < current_end; j++)
			{
				dest_value[j] = inner_list[j - current_prefix];
			}
		}
		return max_count;
	}
	template <typename T>
	inline auto resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& dest, const std::vector<T>& src)
	{
		dest = device.create_buffer<T>(src.size());
	};
	template <typename T>
	inline auto resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& dest, const uint size)
	{
		dest = device.create_buffer<T>(size);
	};
	template <typename T>
	inline auto resize_buffer(luisa::compute::Device& device, std::vector<T>& dest, const uint size)
	{
		dest.resize(size);
	};
	template <typename T>
	inline auto resize_buffer(std::vector<T>& dest, const uint size)
	{
		dest.resize(size);
	};

	template <typename T>
	inline auto get_buffer_MB(std::vector<T>& dest, const uint size)
	{
		return dest.size() * sizeof(T) / (1024.0f * 1024.0f);
	};
	template <typename T>
	inline auto get_buffer_MB(luisa::compute::Buffer<T>& dest, const uint size)
	{
		return dest.buffer_size_bytes() / (1024.0f * 1024.0f);
	};

	template <typename T>
	void dynamic_resize_template(luisa::compute::Device& device,
		luisa::compute::Buffer<T>&						 buffer,
		const uint										 curr_count,
		const std::string_view							 name)
	{
		if (curr_count > buffer.size())
		{
			const uint desired_size = curr_count * 2;

			LUISA_INFO("Resize buffer {} : from {} (< 2*CurrMax = {}) to {} ({} MB)",
				name,
				buffer.size(),
				curr_count * 2,
				desired_size,
				uint(desired_size * sizeof(T) / 1024 / 1024));
			buffer.release();
			buffer = device.create_buffer<T>(desired_size);
		}
	}

	// template<typename BufferType, typename Element>
	// constexpr auto get_buffer_view_type() { return std::enable_if_t<
	//             std::is_same_v<BufferType<Element>, luisa::compute::Buffer<Element>>,
	//             luisa::compute::BufferView<Element>> }

} // namespace lcs::Initializer