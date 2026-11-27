/* cpu.c: the L7 search of mass8.c for one G7, called by mass8g.cu for the G7 that the GPU found positive.
 * It prints the outputs exactly as mass8 does (emit() in mass8.c) and returns their number. */
#define main mass8_main
#include "mass8.c"
long cpu_g7(const unsigned char *g, long cap) {
    for (int x = 0; x < 128; x++) G7[x] = g[x];
    images(G7, 7, Im7); cosets(Im7, 2);
    nodes = 0; L7[0] = 0; cap_left = cap;
    long before = out_count; bt0();
    return out_count - before;
}
