#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

// Minimal repro of the cross-dispatch staging-reuse race.
// N small buffers, each uploaded in its own `stream << ...;` dispatch with no
// intermediate sync; each carries a per-buffer sentinel uint3(b,0,b).
// On unpatched LuisaCompute (cuStreamWriteValue64 signal) this intermittently
// corrupts buffers; with the cuLaunchHostFunc fix it is clean.
int main(int argc, char** argv)
{
	Context ctx{ std::string{ argv[0] } };
	Device	device = ctx.create_device("cuda", nullptr, true);
	Stream	stream = device.create_stream(StreamTag::COMPUTE);

	constexpr uint N = 500u; // many small buffers stress the staging pool
	constexpr uint M = 4u;	 // tiny each -> dense staging-reuse window

	std::vector<Buffer<uint3>>		buf;
	std::vector<std::vector<uint3>> ref;
	buf.reserve(N);
	ref.reserve(N);
	for (uint b = 0; b < N; b++)
	{
		ref.emplace_back(M, make_uint3(b, 0, b)); // per-buffer sentinel
		buf.emplace_back(device.create_buffer<uint3>(M));
	}
	// cross-dispatch: one statement per buffer -> one dispatch each, no sync between
	for (uint b = 0; b < N; b++)
		stream << buf[b].copy_from(ref[b].data());
	stream << synchronize();

	uint			   bad = 0;
	std::vector<uint3> tmp(M);
	for (uint b = 0; b < N; b++)
	{
		stream << buf[b].copy_to(tmp.data()) << synchronize();
		for (uint i = 0; i < M; i++)
			if (tmp[i].x != b)
			{
				LUISA_WARNING_WITH_LOCATION("CORRUPT buf={} elem={} got=({},{},{}) expected=({},{},{})",
					b, i, tmp[i].x, tmp[i].y, tmp[i].z, b, 0, b);
				bad++;
				break;
			}
	}
	LUISA_INFO("corrupt = {} / {}", bad, N);
	return bad ? 1 : 0;
}