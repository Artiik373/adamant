// src/models/inkling2/walk.cpp - the row buffers of each visit of one forward (see walk.hpp).
#include "models/inkling2/walk.hpp"

#include <algorithm>

namespace basalt::models::inkling2 {

bool walk(const Geometry& g, Walk& w, std::string& err, bool hidden) {
    if (g.block <= 0 || g.loop_start != g.block || g.loop_end - g.loop_start != 2 * g.block ||
        g.n_layers - g.loop_end != g.block || g.loop_passes < 1) {
        err = "the forward takes a prelude of one block, a loop of two blocks and a coda of one block";
        return false;
    }
    w = Walk{};
    w.weak = g.strong_visits() < (int) g.visits.size();
    if (w.weak && g.loop_passes < 2) {
        err = "LoopCD-Logits takes a loop of two passes or more";
        return false;
    }
    const bool keep = (hidden || w.weak) && g.loop_passes > 1;
    std::vector<Buf> blocks, pre_blocks, coda_blocks;
    Buf hid = Buf::X0;
    const auto mid = [](int pass) { return pass % 2 ? Buf::I12b : Buf::I12a; };
    // the head's input of a coda: its last output and the finished blocks
    const auto finish = [&](Buf& fin, int& n, Buf* out) {
        if (blocks.size() + 1 > (size_t) kMaxMix) return false;
        fin = hid;
        n = (int) blocks.size();
        std::copy(blocks.begin(), blocks.end(), out);
        return true;
    };
    for (size_t i = 0; i < g.visits.size(); ++i) {
        const Visit& v = g.visits[i];
        const int l = v.layer;
        if (v.weak && !g.visits[i - 1].weak) {   // the weak coda: the strong one's blocks, the first pass's pair for the last
            if (!finish(w.final_hid, w.n_blocks, w.blocks)) break;
            blocks = coda_blocks;
            blocks.back() = Buf::F12;
            hid = Buf::FOut;
        }
        if (l == g.loop_start) {
            if (v.pass == 0) pre_blocks = blocks;   // every pass starts from the blocks the prelude finished
            else blocks = pre_blocks;
        }
        if (!v.weak && l == g.loop_end) coda_blocks = blocks;
        Step s;
        s.visit = (int) i;
        s.hid = hid;
        s.n_blocks = (int) blocks.size();
        std::copy(blocks.begin(), blocks.end(), s.blocks);
        if (v.pass > 0) {
            s.msgs = true;
            s.bank[0] = mid(v.pass - 1);
            s.bank[1] = Buf::BankOut;
            s.keep_first = keep && v.pass == 1 && l == g.loop_start;
        }
        if (hidden && g.loop_passes > 1 && !v.weak && l == g.loop_end && hid == Buf::I18) {
            s.loopcd = true;
            s.cd[0] = mid(g.loop_passes - 1);
            s.cd[1] = Buf::I18;
        }
        // the output goes where the next visit keeps it
        const Visit* nx = i + 1 < g.visits.size() && g.visits[i + 1].weak == v.weak ? &g.visits[i + 1] : nullptr;
        if (v.weak) s.out = hid == Buf::W0 ? Buf::W1 : Buf::W0;
        else if (nx && nx->layer == g.loop_start) s.out = nx->pass == 0 ? Buf::Pre : Buf::BankOut;
        else if (nx && nx->layer == g.loop_start + g.block) s.out = mid(nx->pass);
        else if (nx && nx->layer == g.loop_end) s.out = Buf::I18;
        else s.out = hid == Buf::H0 ? Buf::H1 : Buf::H0;
        w.steps.push_back(s);
        if (g.boundary(l)) blocks.push_back(hid);   // the layer's input joins the finished blocks
        // a later pass reads the prelude's output again, not the previous pass's
        hid = nx && nx->layer == g.loop_start ? Buf::Pre : s.out;
    }
    if (w.steps.size() != g.visits.size() ||
        !(w.weak ? finish(w.weak_final, w.n_weak_blocks, w.weak_blocks) : finish(w.final_hid, w.n_blocks, w.blocks))) {
        err = "more finished blocks than a mix reads";
        return false;
    }
    return true;
}

std::vector<Segment> split_walk(const Geometry& g, const Walk& w, const std::vector<int>& card_of) {
    std::vector<Segment> segs;
    uint8_t valid[kBufs] = {};   // the cards holding each buffer's last write
    valid[(int) Buf::X0] = 1;    // the embedding, on the first card
    int msg_pass[2] = {-1, -1};  // the pass each card's messages were last mixed for
    auto need = [&](Segment& sg, Buf b) {
        if (!(valid[(int) b] & (1u << sg.card))) {
            sg.pull.push_back(b);
            valid[(int) b] |= (uint8_t) (1u << sg.card);
        }
    };
    for (size_t si = 0; si < w.steps.size(); ++si) {
        const Step& s = w.steps[si];
        const Visit& v = g.visits[(size_t) s.visit];
        const int card = card_of[(size_t) v.layer];
        if (segs.empty() || segs.back().card != card) segs.push_back(Segment{card, si, si, false, {}, {}});
        Segment& sg = segs.back();
        const bool mix = s.msgs && msg_pass[card] != v.pass;
        if (mix || s.keep_first) {
            if (mix) msg_pass[card] = v.pass;
            need(sg, s.bank[0]);
            need(sg, s.bank[1]);
        }
        if (s.keep_first) valid[(int) Buf::F12] = valid[(int) Buf::FOut] = (uint8_t) (1u << card);
        if (s.loopcd) {   // the extrapolated pair is this card's alone
            for (Buf b : {Buf::F12, Buf::FOut, s.cd[0], s.cd[1]}) need(sg, b);
            valid[(int) s.cd[0]] = valid[(int) s.cd[1]] = (uint8_t) (1u << card);
        }
        need(sg, s.hid);
        for (int i = 0; i < s.n_blocks; ++i) need(sg, s.blocks[i]);
        sg.mix_msgs.push_back(mix ? 1 : 0);
        sg.s1 = si + 1;
        valid[(int) s.out] = (uint8_t) (1u << card);
    }
    if (segs.empty() || segs.back().card != 0) segs.push_back(Segment{0, w.steps.size(), w.steps.size(), false, {}, {}});
    Segment& hs = segs.back();
    hs.head = true;
    for (int i = 0; i < w.n_blocks; ++i) need(hs, w.blocks[i]);
    need(hs, w.final_hid);
    if (w.weak) {   // LoopCD-Logits: the weak coda's head too
        for (int i = 0; i < w.n_weak_blocks; ++i) need(hs, w.weak_blocks[i]);
        need(hs, w.weak_final);
    }
    return segs;
}

int split_handoffs(const Geometry& g, int cut) {
    Walk w;
    std::string err;
    if (!walk(g, w, err)) return 0;
    std::vector<int> card_of((size_t) g.n_layers, 0);
    for (int l = cut > 0 ? cut : g.n_layers; l < g.n_layers; ++l) card_of[(size_t) l] = 1;
    return (int) split_walk(g, w, card_of).size() - 1;
}

}  // namespace basalt::models::inkling2
