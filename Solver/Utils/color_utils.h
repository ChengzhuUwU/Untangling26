#pragma once

#include <array>
#include <vector>

namespace lcs
{
	namespace ColorUtils
	{
		inline std::vector<std::array<float, 3>> camp10()
		{
			return { { 0.12156862745098039, 0.4666666666666667, 0.7058823529411765 },
				{ 1.0, 0.4980392156862745, 0.054901960784313725 },
				{ 0.17254901960784313, 0.6274509803921569, 0.17254901960784313 },
				{ 0.8392156862745098, 0.15294117647058825, 0.1568627450980392 },
				{ 0.5803921568627451, 0.403921568627451, 0.7411764705882353 },
				{ 0.5490196078431373, 0.33725490196078434, 0.29411764705882354 },
				{ 0.8901960784313725, 0.4666666666666667, 0.7607843137254902 },
				{ 0.4980392156862745, 0.4980392156862745, 0.4980392156862745 },
				{ 0.7372549019607844, 0.7411764705882353, 0.13333333333333333 },
				{ 0.09019607843137255, 0.7450980392156863, 0.8117647058823529 } };
		}
		inline std::array<float, 3> camp10(const uint idx)
		{
			return camp10()[idx % 10];
		}
		inline std::vector<std::array<float, 3>> highlight10()
		{
			return {
				{ 1.0f, 0.0f, 0.0f }, // Red
				{ 0.0f, 1.0f, 0.0f }, // Green
				{ 0.0f, 0.0f, 1.0f }, // Blue
				{ 1.0f, 1.0f, 0.0f }, // Yellow
				{ 1.0f, 0.0f, 1.0f }, // Magenta
				{ 0.0f, 1.0f, 1.0f }, // Cyan
				{ 1.0f, 0.5f, 0.0f }, // Orange
				{ 0.5f, 0.0f, 1.0f }, // Purple
				{ 0.0f, 0.5f, 1.0f }, // Sky Blue
				{ 0.5f, 1.0f, 0.0f }  // Lime Green
			};
		}
		inline std::array<float, 3> highlight10(const uint idx)
		{
			return highlight10()[idx % 10];
		}

		inline std::array<float, 3> hsvToRgb(float h, float s, float v)
		{
			// h: 0-360, s: 0-1, v: 0-1
			float c = v * s;
			float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
			float m = v - c;
			float r, g, b;
			if (h < 60)
			{
				r = c;
				g = x;
				b = 0;
			}
			else if (h < 120)
			{
				r = x;
				g = c;
				b = 0;
			}
			else if (h < 180)
			{
				r = 0;
				g = c;
				b = x;
			}
			else if (h < 240)
			{
				r = 0;
				g = x;
				b = c;
			}
			else if (h < 300)
			{
				r = x;
				g = 0;
				b = c;
			}
			else
			{
				r = c;
				g = 0;
				b = x;
			}
			return { r + m, g + m, b + m };
		}

		inline std::array<float, 3> mapFloatToColor_HSV(float alpha)
		{
			alpha = std::fmax(0.0f, std::fmin(1.0f, alpha));
			float hue = 240.0f * (1.0f - alpha); // 240 -> 0
			return hsvToRgb(hue, 1.0f, 1.0f);
		}

	} // namespace ColorUtils

}; // namespace lcs