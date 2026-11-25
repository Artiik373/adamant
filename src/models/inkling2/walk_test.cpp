// src/models/inkling2/walk_test.cpp - the walk against the forward it replaces: a forward that copies each kept input into
// a buffer of its own. Both are run over symbols (each visit's output a distinct value); every input, block and bank entry
// a step reads must hold the value that forward reads there.
#include "models/inkling2/walk.hpp"

#include <cstdio>
#include <vector>

using namespace basalt::models::inkling2;

namespace {

int failures = 0;
void expect(bool ok, const char* what, int visit) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s at visit %d\n", what, visit);
        ++failures;
    }
}

constexpr int kEmb = -1;   // the embedding's symbol; visit i's output is i

struct Reads {
    int hid = 0;
    std::vector<int> blocks;
    bool msgs = false;
    int bank[2] = {0, 0};
};

// The copying forward: the prelude's output in its own buffer, the bank's two entries, the kept inputs.
std::vector<Reads> reference(const Geometry& g, int& final_hid, std::vector<int>& final_blocks) {
    std::vector<Reads> out;
    std::vector<int> blocks, pre_blocks;
    int hid = kEmb, pre = 0, bank_out = 0, i12[2] = {0, 0}, bank = 0;
    for (size_t i = 0; i < g.visits.size(); ++i) {
        const Visit& v = g.visits[i];
        const int l = v.layer;
        if (l == g.loop_start) {
            if (v.pass == 0) {
                pre = hid;
                pre_blocks = blocks;
            } else {
                bank_out = hid;
                bank = i12[(v.pass - 1) % 2];
                blocks = pre_blocks;
            }
            hid = pre;
        }
        Reads r;
        r.hid = hid;
        r.blocks = blocks;
        if (v.pass > 0) {
            r.msgs = true;
            r.bank[0] = bank;
            r.bank[1] = bank_out;
        }
        out.push_back(r);
        if (g.boundary(l)) {
            if (g.looped(l) && l != g.loop_start) i12[v.pass % 2] = hid;
            blocks.push_back(hid);
        }
        hid = (int) i;
    }
    final_hid = hid;
    final_blocks = blocks;
    return out;
}

void check(int layers, int block, int passes, bool loopcd = false) {
    Geometry g;
    g.n_layers = layers;
    g.block = block;
    g.loop_start = block;
    g.loop_end = 3 * block;
    g.loop_passes = passes;
    g.visits = loop_visits(layers, g.loop_start, g.loop_end, passes);
    Walk w;
    std::string err;
    if (!walk(g, w, err, loopcd)) {
        std::fprintf(stderr, "FAIL walk: %s\n", err.c_str());
        ++failures;
        return;
    }
    int ref_final = 0;
    std::vector<int> ref_blocks;
    const std::vector<Reads> ref = reference(g, ref_final, ref_blocks);
    expect(w.steps.size() == g.visits.size(), "a step per visit", 0);
    int buf[kBufs];
    for (int& b : buf) b = -2;   // nothing written yet
    buf[(int) Buf::X0] = kEmb;
    int keeps = 0, cds = 0, keep_at = -1;
    for (size_t i = 0; i < w.steps.size() && i < ref.size(); ++i) {
        const Step& s = w.steps[i];
        const int vi = (int) i;
        if (s.keep_first) {   // LoopCD: the first pass's pair is the second pass's bank
            ++keeps;
            keep_at = vi;
            expect(s.msgs && g.visits[i].pass == 1 && g.visits[i].layer == g.loop_start, "kept at the second pass's start", vi);
            buf[(int) Buf::F12] = buf[(int) s.bank[0]];
            buf[(int) Buf::FOut] = buf[(int) s.bank[1]];
        }
        if (s.loopcd) {   // the last pass's pair, against the first's, before the coda
            ++cds;
            expect(g.visits[i].layer == g.loop_end && keep_at >= 0, "LoopCD at the coda's start", vi);
            if (keep_at >= 0) {
                expect(buf[(int) Buf::F12] == ref[(size_t) keep_at].bank[0], "LoopCD first block input", vi);
                expect(buf[(int) Buf::FOut] == ref[(size_t) keep_at].bank[1], "LoopCD first output", vi);
            }
            expect(!ref[i].blocks.empty() && buf[(int) s.cd[0]] == ref[i].blocks.back(), "LoopCD last block input", vi);
            expect(buf[(int) s.cd[1]] == ref[i].hid && s.cd[1] == s.hid, "LoopCD last output", vi);
        }
        expect(s.visit == vi, "visit order", vi);
        expect(buf[(int) s.hid] == ref[i].hid, "input", vi);
        expect(s.n_blocks == (int) ref[i].blocks.size(), "block count", vi);
        for (int j = 0; j < s.n_blocks && j < (int) ref[i].blocks.size(); ++j)
            expect(buf[(int) s.blocks[j]] == ref[i].blocks[(size_t) j], "block", vi);
        expect(s.msgs == ref[i].msgs, "messages", vi);
        if (s.msgs) {
            expect(buf[(int) s.bank[0]] == ref[i].bank[0], "bank entry 0", vi);
            expect(buf[(int) s.bank[1]] == ref[i].bank[1], "bank entry 1", vi);
        }
        expect(s.out != s.hid, "output over its own input", vi);
        expect(s.out != Buf::F12 && s.out != Buf::FOut, "output over a kept pair", vi);
        buf[(int) s.out] = vi;
    }
    const int want = loopcd && passes > 1 ? 1 : 0;
    expect(keeps == want && cds == want, "LoopCD steps", -1);
    expect(buf[(int) w.final_hid] == ref_final, "the head's input", -1);
    expect(w.n_blocks == (int) ref_blocks.size(), "the head's block count", -1);
    for (int j = 0; j < w.n_blocks && j < (int) ref_blocks.size(); ++j)
        expect(buf[(int) w.blocks[j]] == ref_blocks[(size_t) j], "the head's block", -1);
}

// LoopCD-Logits: the strong forward reads as without the weak coda, and the weak coda is the coda run from the first
// pass's pair (its input the first pass's output, the last finished block the first pass's input of the second block).
void check_weak(int layers, int block, int passes, const std::vector<int>& card_of = {}) {
    Geometry g;
    g.n_layers = layers;
    g.block = block;
    g.loop_start = block;
    g.loop_end = 3 * block;
    g.loop_passes = passes;
    g.visits = loop_visits(layers, g.loop_start, g.loop_end, passes);
    int ref_final = 0;
    std::vector<int> ref_blocks;
    const std::vector<Reads> ref = reference(g, ref_final, ref_blocks);
    const size_t n_strong = g.visits.size();
    add_weak_coda(g);
    Walk w;
    std::string err;
    const bool ok = walk(g, w, err);
    expect(ok == (passes > 1), "LoopCD-Logits needs two passes", passes);
    if (!ok) return;
    expect(w.weak && w.steps.size() == g.visits.size() && g.strong_visits() == (int) n_strong, "the weak steps", -1);
    // the buffers of each card (one card when card_of is empty), and the value each holds on the one-card walk
    const std::vector<Segment> segs = card_of.empty() ? split_walk(g, w, std::vector<int>((size_t) layers, 0)) : split_walk(g, w, card_of);
    int one[kBufs], two[2][kBufs];
    for (int b = 0; b < kBufs; ++b) one[b] = two[0][b] = two[1][b] = -2;
    one[(int) Buf::X0] = two[0][(int) Buf::X0] = kEmb;
    int first_mid = -9, first_out = -9, weak_hid = -9;
    std::vector<int> weak_blocks;
    for (const Segment& sg : segs) {
        for (Buf b : sg.pull) two[sg.card][(int) b] = two[1 - sg.card][(int) b];
        int* mine = two[sg.card];
        for (size_t i = sg.s0; i < sg.s1; ++i) {
            const Step& s = w.steps[i];
            const Visit& v = g.visits[i];
            const int vi = (int) i;
            expect(mine[(int) s.hid] == one[(int) s.hid], "weak split input", vi);
            for (int j = 0; j < s.n_blocks; ++j) expect(mine[(int) s.blocks[j]] == one[(int) s.blocks[j]], "weak split block", vi);
            if (s.keep_first) {
                first_mid = ref[i].bank[0];
                first_out = ref[i].bank[1];
                one[(int) Buf::F12] = mine[(int) Buf::F12] = one[(int) s.bank[0]];
                one[(int) Buf::FOut] = mine[(int) Buf::FOut] = one[(int) s.bank[1]];
            }
            if (!v.weak) {
                expect(one[(int) s.hid] == ref[i].hid, "strong input", vi);
                for (int j = 0; j < s.n_blocks && j < (int) ref[i].blocks.size(); ++j)
                    expect(one[(int) s.blocks[j]] == ref[i].blocks[(size_t) j], "strong block", vi);
            } else {
                if (!g.visits[i - 1].weak) {   // the weak coda's start, from the strong coda's start
                    weak_hid = first_out;
                    weak_blocks = ref[n_strong - (size_t) (layers - g.loop_end)].blocks;
                    weak_blocks.back() = first_mid;
                }
                expect(one[(int) s.hid] == weak_hid, "weak input", vi);
                expect(s.n_blocks == (int) weak_blocks.size(), "weak block count", vi);
                for (int j = 0; j < s.n_blocks && j < (int) weak_blocks.size(); ++j)
                    expect(one[(int) s.blocks[j]] == weak_blocks[(size_t) j], "weak block", vi);
                expect(!s.msgs, "no messages in the weak coda", vi);
                if (g.boundary(v.layer)) weak_blocks.push_back(weak_hid);
                weak_hid = vi;
            }
            one[(int) s.out] = mine[(int) s.out] = vi;
        }
        if (sg.head) {
            int* mine = two[sg.card];
            expect(mine[(int) w.final_hid] == ref_final && one[(int) w.final_hid] == ref_final, "the strong head's input", -1);
            expect(w.n_blocks == (int) ref_blocks.size(), "the strong head's block count", -1);
            for (int j = 0; j < w.n_blocks && j < (int) ref_blocks.size(); ++j)
                expect(mine[(int) w.blocks[j]] == ref_blocks[(size_t) j], "the strong head's block", -1);
            expect(mine[(int) w.weak_final] == weak_hid, "the weak head's input", -1);
            expect(w.n_weak_blocks == (int) weak_blocks.size(), "the weak head's block count", -1);
            for (int j = 0; j < w.n_weak_blocks && j < (int) weak_blocks.size(); ++j)
                expect(mine[(int) w.weak_blocks[j]] == weak_blocks[(size_t) j], "the weak head's block", -1);
        }
    }
}

// The split walk over two cards: each card's buffers hold what its own steps wrote and what its segments pulled; every
// read on a card must find the value the one-card walk reads, and a later pass's messages must be mixed on the card first.
void check_split(int cut, bool loopcd = false) {
    Geometry g;
    g.n_layers = 24;
    g.block = 6;
    g.loop_start = 6;
    g.loop_end = 18;
    g.loop_passes = 3;
    g.visits = loop_visits(24, 6, 18, 3);
    Walk w;
    std::string err;
    if (!walk(g, w, err, loopcd)) return;
    std::vector<int> card_of(24, 0);
    for (int l = cut; l < 24; ++l) card_of[(size_t) l] = 1;
    const std::vector<Segment> segs = split_walk(g, w, card_of);
    int one[kBufs], two[2][kBufs];
    for (int b = 0; b < kBufs; ++b) one[b] = two[0][b] = two[1][b] = -2;
    one[(int) Buf::X0] = two[0][(int) Buf::X0] = kEmb;
    int mixed[2] = {-1, -1};
    size_t next = 0;
    for (size_t k = 0; k < segs.size(); ++k) {
        const Segment& sg = segs[k];
        expect(sg.s0 == next && sg.s1 >= sg.s0, "segments in order", (int) k);
        expect(k + 1 == segs.size() ? sg.head && sg.card == 0 : !sg.head, "the head on the first card, last", (int) k);
        for (Buf b : sg.pull) two[sg.card][(int) b] = two[1 - sg.card][(int) b];
        int* mine = two[sg.card];
        for (size_t i = sg.s0; i < sg.s1; ++i) {
            const Step& s = w.steps[i];
            const int vi = (int) i, pass = g.visits[i].pass;
            if (s.keep_first) {
                expect(mine[(int) s.bank[0]] == one[(int) s.bank[0]] && mine[(int) s.bank[1]] == one[(int) s.bank[1]], "split kept pair", vi);
                one[(int) Buf::F12] = mine[(int) Buf::F12] = one[(int) s.bank[0]];
                one[(int) Buf::FOut] = mine[(int) Buf::FOut] = one[(int) s.bank[1]];
            }
            if (s.loopcd) {   // its inputs on this card; the other card's copies of the pair go stale
                for (Buf b : {Buf::F12, Buf::FOut, s.cd[0], s.cd[1]}) expect(mine[(int) b] == one[(int) b], "split LoopCD input", vi);
                two[1 - sg.card][(int) s.cd[0]] = two[1 - sg.card][(int) s.cd[1]] = -3;
            }
            expect(card_of[(size_t) g.visits[i].layer] == sg.card, "a step on its layer's card", vi);
            expect(mine[(int) s.hid] == one[(int) s.hid], "split input", vi);
            for (int j = 0; j < s.n_blocks; ++j) expect(mine[(int) s.blocks[j]] == one[(int) s.blocks[j]], "split block", vi);
            if (sg.mix_msgs[i - sg.s0]) {
                expect(s.msgs, "messages mixed on a step that takes them", vi);
                expect(mine[(int) s.bank[0]] == one[(int) s.bank[0]] && mine[(int) s.bank[1]] == one[(int) s.bank[1]], "split bank", vi);
                mixed[sg.card] = pass;
            }
            if (s.msgs) expect(mixed[sg.card] == pass, "this pass's messages on the card", vi);
            one[(int) s.out] = mine[(int) s.out] = vi;
        }
        next = sg.s1;
        if (sg.head) {
            expect(mine[(int) w.final_hid] == one[(int) w.final_hid], "split head input", -1);
            for (int j = 0; j < w.n_blocks; ++j) expect(mine[(int) w.blocks[j]] == one[(int) w.blocks[j]], "split head block", -1);
        }
    }
    expect(next == w.steps.size(), "every step in a segment", -1);
    // one card: one segment; a cut at a loop edge: there and back; inside the loop: three crossings a pass
    const int h = split_handoffs(g, cut);
    expect(h == (cut <= 6 || cut >= 18 ? 2 : 6), "hand-offs", cut);
}

}  // namespace

int main() {
    for (int cut = 1; cut < 24; ++cut) {
        check_split(cut);
        check_split(cut, true);
    }
    {
        Geometry g;
        g.n_layers = 24;
        g.block = 6;
        g.loop_start = 6;
        g.loop_end = 18;
        g.loop_passes = 3;
        g.visits = loop_visits(24, 6, 18, 3);
        Walk w;
        std::string err;
        walk(g, w, err);
        const std::vector<Segment> one = split_walk(g, w, std::vector<int>(24, 0));
        expect(one.size() == 1 && one[0].head && one[0].pull.empty() && split_handoffs(g, 0) == 0, "one card, one segment", -1);
        for (size_t i = 0; i < w.steps.size(); ++i)   // one card mixes a pass's messages at the loop's first layer, as before
            expect(one[0].mix_msgs[i] == (w.steps[i].msgs && g.visits[i].layer == g.loop_start), "one card's message mixes", (int) i);
    }
    check(24, 6, 3);   // Inkling 2 Mini
    check(24, 6, 1);
    check(24, 6, 2);
    check(24, 6, 4);
    check(8, 2, 5);
    for (int passes : {1, 2, 3, 4}) {
        check(24, 6, passes, true);
        check_weak(24, 6, passes);
    }
    check_weak(8, 2, 5);
    for (int cut = 1; cut < 24; ++cut) {
        std::vector<int> card_of(24, 0);
        for (int l = cut; l < 24; ++l) card_of[(size_t) l] = 1;
        check_weak(24, 6, 3, card_of);
    }
    check(8, 2, 5, true);
    {   // a layout the walk does not cover
        Geometry g;
        g.n_layers = 24;
        g.block = 6;
        g.loop_start = 6;
        g.loop_end = 12;
        g.loop_passes = 2;
        g.visits = loop_visits(24, 6, 12, 2);
        Walk w;
        std::string err;
        expect(!walk(g, w, err), "refuses a loop of one block", -1);
    }
    if (failures) return 1;
    std::printf("inkling2 walk: ok\n");
    return 0;
}
