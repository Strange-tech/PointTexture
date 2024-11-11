#include <fstream>
#include <iostream>
using namespace std;

int main()
{
    int origin[2] = {0, 0};
    int width = 264, height = 418;

    int window_origin[2] = {11, -28};
    int window_width = 24, window_height = 38;
    int seperate_width = 54, seperate_height = 83;
    int repeat_h = 5, repeat_v = 5;

    int total_points, total_edges;
    // total_points = total_edges = (repeat_h * repeat_v + 1) * 4;
    total_points = total_edges = repeat_h * repeat_v * 4;

    ofstream file("wall.txt");
    if(file.is_open())
    {
        // write points
        file << total_points << ' ' << total_edges << '\n';
        // file << origin[0] << ' ' << origin[1] << '\n';
        // file << origin[0] + width << ' ' << origin[1] << '\n';
        // file << origin[0] + width << ' ' << origin[1] - height << '\n';
        // file << origin[0] << ' ' << origin[1] - height << '\n';

        for(int i = 0; i < repeat_h; i++)
        {
            for(int j = 0; j < repeat_v; j++)
            {
                int w_o_h = window_origin[0] + i * seperate_width;
                int w_o_v = window_origin[1] - j * seperate_height;
                file << w_o_h << ' ' << w_o_v << '\n';
                file << w_o_h + window_width << ' ' << w_o_v << '\n';
                file << w_o_h + window_width << ' ' << w_o_v - window_height
                     << '\n';
                file << w_o_h << ' ' << w_o_v - window_height << '\n';
            }
        }
        // write edges
        for(int i = 0; i < total_points; i += 4)
        {
            file << i << ' ' << i + 1 << '\n';
            file << i + 1 << ' ' << i + 2 << '\n';
            file << i + 2 << ' ' << i + 3 << '\n';
            file << i + 3 << ' ' << i << '\n';
        }
        file.close();
    }

    return 0;
}