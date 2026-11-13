#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10, place by place, for the scored window.

Approach: each game seen in the history is modelled on its own as a
distribution over log-rank that is pushed forward one hour at a time from the
last read to the end of the window.  Every hour the distribution is shifted by
a decaying momentum drift and a pull toward the game's own typical rank,
spread by a rank-dependent log-rank diffusion fitted from the history itself,
and drained by a rank-dependent exit hazard (games leave the chart) with a
small re-entry flow for absent games.  Games whose creation date is known get
an extra exit hazard as their 28-day chart eligibility ends.  The hourly
place masses are averaged over the window, because one forecast is scored
against every hourly read: the honest forecast is the time-averaged occupancy
distribution.  Each place's distribution is normalised to leave a small
unnamed reserve for games never seen before.

Standard library only.  No network, no model calls, no randomness: the same
input always writes the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

# ---- TUNABLE PARAMETERS (begin) ----
MOMENTUM = 0.3
MOMENTUM_HALFLIFE_H = 12.0
MEAN_REVERSION = 0.0
EXIT_SCALE = 1.0
RETURN_SCALE = 1.0
RESERVE = 0.2
TOPN_PER_PLACE = 60
MIN_ROW_P = 0.0005
MAX_GAMES = 160
GRID_STEP = 0.05
GRID_LO = -0.6931471805599453
GRID_HI = 4.867534450455582
SIGMA_DEFAULT = 0.09
SIGMA_SCALE = 1.6
SIGMA_MIN = 0.04
SIGMA_MAX = 0.2
EXIT_DEFAULT = 0.004
EXIT_MIN = 0.0002
EXIT_MAX = 0.05
RETURN_DEFAULT = 0.002
RETURN_MAX = 0.01
AGE_RAMP_HOURS = 24.0
AGE_HAZARD = 0.1
AGE_POST_HAZARD = 0.5
# ---- TUNABLE PARAMETERS (end) ----

FLOOR_P = 0.0001
CHART_DAYS = 28.0
SECONDS_PER_DAY = 86400.0


def parse_time(text):
    """Parse an ISO-8601 timestamp to epoch seconds; None when unparsable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        cleaned = text[:-1] + "+00:00" if text.endswith("Z") else text
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def read_history(input_dir):
    """Read the snapshot lines, oldest first, keeping usable reads only."""
    path = os.path.join(input_dir, "history.ndjson")
    reads = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or record.get("ok") is False:
                continue
            chart = record.get("up_and_coming")
            if not isinstance(chart, list) or not chart:
                continue
            stamp = parse_time(record.get("at"))
            if stamp is None:
                continue
            rows = []
            seen = set()
            for row in chart:
                if not isinstance(row, dict):
                    continue
                try:
                    uid = int(row["universe_id"])
                    rank = int(row["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
                if rank < 1 or uid in seen:
                    continue
                seen.add(uid)
                rows.append((rank, uid))
            if not rows:
                continue
            rows.sort()
            reads.append({
                "t": stamp,
                "ranks": dict((uid, rank) for rank, uid in rows),
                "order": [uid for _, uid in rows],
                "details": record.get("details") if isinstance(record.get("details"), dict) else {},
            })
    reads.sort(key=lambda r: r["t"])
    return reads


def collect_created(reads):
    """Map universe id -> creation epoch seconds, from details when present."""
    created = {}
    for read in reads:
        for key, info in read["details"].items():
            if not isinstance(info, dict):
                continue
            try:
                uid = int(key)
            except (TypeError, ValueError):
                continue
            if uid in created:
                continue
            stamp = parse_time(info.get("created"))
            if stamp is not None:
                created[uid] = stamp
    return created


def age_signal_trustworthy(reads, created, last_t):
    """Only trust creation dates if no game is ever seen long past its
    nominal 28-day expiry; otherwise 'created' is not the publish date."""
    limit = CHART_DAYS * SECONDS_PER_DAY + 36.0 * 3600.0
    for read in reads:
        for uid in read["ranks"]:
            born = created.get(uid)
            if born is not None and read["t"] - born > limit:
                return False
    return True


def fit_rates(reads):
    """Fit diffusion sigmas and exit/return hazards from the history.

    Sigma per rank bucket is the spread of log-rank moves scaled by the
    square root of the gap; hazards are events per game-hour.  Bucket
    estimates shrink toward the global value so short histories stay sane.
    """
    buckets = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 10 ** 9)]
    centers = [10.0, 30.0, 50.0, 70.0, 90.0]

    def bucket_of(rank):
        for i, (lo, hi) in enumerate(buckets):
            if lo <= rank <= hi:
                return i
        return len(buckets) - 1

    moves = [[] for _ in buckets]
    exits = [0 for _ in buckets]
    exposure = [0.0 for _ in buckets]
    returns = 0
    absent_hours = 0.0
    ever_seen = set()
    for prev, cur in zip(reads, reads[1:]):
        gap = (cur["t"] - prev["t"]) / 3600.0
        if gap < 0.5 or gap > 24.0:
            continue
        for uid, rank in prev["ranks"].items():
            b = bucket_of(rank)
            exposure[b] += gap
            nxt = cur["ranks"].get(uid)
            if nxt is None:
                exits[b] += 1
            else:
                moves[b].append((math.log(nxt) - math.log(rank)) / math.sqrt(gap))
        for uid in ever_seen:
            if uid not in prev["ranks"]:
                absent_hours += gap
                if uid in cur["ranks"]:
                    returns += 1
        ever_seen |= set(prev["ranks"])
    ever_seen |= set(reads[-1]["ranks"]) if reads else set()

    all_moves = [m for group in moves for m in group]
    if len(all_moves) >= 20:
        mean = sum(all_moves) / len(all_moves)
        global_sd = math.sqrt(sum((m - mean) ** 2 for m in all_moves) / len(all_moves))
    else:
        global_sd = SIGMA_DEFAULT
    global_sd = min(max(global_sd, SIGMA_MIN), SIGMA_MAX)

    sigmas = []
    for group in moves:
        if len(group) >= 20:
            mean = sum(group) / len(group)
            sd = math.sqrt(sum((m - mean) ** 2 for m in group) / len(group))
            sd = (len(group) * sd + 30.0 * global_sd) / (len(group) + 30.0)
        else:
            sd = global_sd
        sigmas.append(min(max(sd, SIGMA_MIN), SIGMA_MAX))

    total_exits = sum(exits)
    total_hours = sum(exposure)
    global_hazard = (total_exits / total_hours) if total_hours > 0 else EXIT_DEFAULT
    global_hazard = min(max(global_hazard, EXIT_MIN), EXIT_MAX)
    hazards = []
    for i in range(len(buckets)):
        if exposure[i] > 0:
            rate = (exits[i] + 2000.0 * global_hazard) / (exposure[i] + 2000.0)
        else:
            rate = global_hazard
        hazards.append(min(max(rate, EXIT_MIN), EXIT_MAX))

    if absent_hours > 0:
        ret = (returns + 5000.0 * RETURN_DEFAULT) / (absent_hours + 5000.0)
    else:
        ret = RETURN_DEFAULT
    ret = min(max(ret, 0.0), RETURN_MAX)

    return centers, sigmas, hazards, ret


def interp_in_log_rank(centers, values, rank):
    """Piecewise-linear interpolation over log(rank), flat outside."""
    x = math.log(max(rank, 1.0))
    xs = [math.log(c) for c in centers]
    if x <= xs[0]:
        return values[0]
    if x >= xs[-1]:
        return values[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = (x - xs[i]) / (xs[i + 1] - xs[i])
            return values[i] * (1.0 - frac) + values[i + 1] * frac
    return values[-1]


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class Grid:
    """Log-rank grid with precomputed diffusion kernels and place map."""

    def __init__(self, centers, sigmas):
        self.step = GRID_STEP
        self.lo = GRID_LO
        count = int(math.floor((GRID_HI - GRID_LO) / GRID_STEP)) + 1
        self.points = [GRID_LO + i * GRID_STEP for i in range(count)]
        self.size = count
        self.place_of = []
        for x in self.points:
            rank = math.exp(x)
            place = -1
            for k in range(1, 11):
                if rank < k + 0.5:
                    place = k if rank >= k - 0.5 else -1
                    break
            self.place_of.append(place)
        self.kernels = []
        for x in self.points:
            sigma = interp_in_log_rank(centers, sigmas, math.exp(x)) * SIGMA_SCALE
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
            radius = max(1, int(math.ceil(3.0 * sigma / GRID_STEP)))
            weights = []
            total = 0.0
            for off in range(-radius, radius + 1):
                hi = (off + 0.5) * GRID_STEP / sigma
                lo = (off - 0.5) * GRID_STEP / sigma
                w = normal_cdf(hi) - normal_cdf(lo)
                weights.append((off, w))
                total += w
            self.kernels.append([(off, w / total) for off, w in weights])
        self.exit_hazard = None

    def set_hazards(self, centers, hazards):
        self.exit_hazard = [
            interp_in_log_rank(centers, hazards, math.exp(x)) for x in self.points
        ]


def build_game_states(reads, centers, sigmas, hazards, ret_hazard, created, trust_age, last_t):
    """Initial per-game distributions plus per-game drift anchors."""
    grid_n = None
    last = reads[-1]
    recent = reads[-3:]
    present_recent = set()
    for read in recent:
        present_recent |= set(read["ranks"])

    series = {}
    for read in reads:
        for uid, rank in read["ranks"].items():
            series.setdefault(uid, []).append((read["t"], rank))

    def velocity(uid):
        obs = series.get(uid, [])
        if len(obs) < 2:
            return 0.0
        window = obs[-4:]
        t0, r0 = window[0]
        t1, r1 = window[-1]
        span = (t1 - t0) / 3600.0
        if span < 3.0:
            return 0.0
        raw = (math.log(r1) - math.log(r0)) / span
        damp = min(1.0, len(window) / 3.0)
        return max(-0.06, min(0.06, raw * damp))

    def anchor(uid):
        obs = series.get(uid, [])
        if len(obs) < 3:
            return math.log(obs[-1][1]) if obs else math.log(50.0)
        weighted = []
        for t, rank in obs:
            age_h = (last_t - t) / 3600.0
            weighted.append((math.log(rank), 0.5 ** (age_h / 72.0)))
        weighted.sort()
        total = sum(w for _, w in weighted)
        acc = 0.0
        for value, w in weighted:
            acc += w
            if acc >= 0.5 * total:
                return value
        return weighted[-1][0]

    def potential(uid):
        obs = series.get(uid, [])
        best = min(rank for _, rank in obs[-5:]) if obs else 999
        vel = velocity(uid)
        score = best - max(0.0, -vel) * 400.0
        if uid not in last["ranks"]:
            score += 40.0
        return score

    candidates = sorted(present_recent | set(last["ranks"]), key=lambda u: (potential(u), u))
    candidates = candidates[:MAX_GAMES]
    states = []
    for uid in candidates:
        rank = last["ranks"].get(uid)
        expiry = None
        if trust_age and uid in created:
            expiry = created[uid] + CHART_DAYS * SECONDS_PER_DAY
        states.append({
            "uid": uid,
            "rank": rank,
            "velocity": velocity(uid) if rank is not None else 0.0,
            "anchor": anchor(uid),
            "expiry": expiry,
        })
    return states


def propagate(grid, states, hours, ret_hazard, last_t):
    """Push every game's distribution forward hour by hour; return the
    time-averaged probability of each game at each place."""
    size = grid.size
    step = grid.step
    lo = grid.lo
    hi = grid.points[-1]
    place_of = grid.place_of
    kernels = grid.kernels
    exit_hazard = grid.exit_hazard

    totals = {}
    live = []
    for st in states:
        dist = [0.0] * size
        absent = 0.0
        if st["rank"] is not None:
            idx = int(round((math.log(st["rank"]) - lo) / step))
            idx = min(max(idx, 0), size - 1)
            dist[idx] = 1.0
        else:
            absent = 1.0
        live.append([st, dist, absent])
        totals[st["uid"]] = [0.0] * 11

    entry_weights = []
    entry_total = 0.0
    for x in grid.points:
        w = math.exp(-0.5 * ((x - math.log(85.0)) / 0.4) ** 2)
        entry_weights.append(w)
        entry_total += w
    entry_weights = [w / entry_total for w in entry_weights]

    for hour in range(1, hours + 1):
        now = last_t + hour * 3600.0
        decay = 0.5 ** ((hour - 1) / MOMENTUM_HALFLIFE_H)
        for item in live:
            st, dist, absent = item
            drift = st["velocity"] * MOMENTUM * decay
            shifted = [0.0] * size
            overflow = 0.0
            for j, mass in enumerate(dist):
                if mass <= 0.0:
                    continue
                target = grid.points[j] + drift - MEAN_REVERSION * (grid.points[j] - st["anchor"])
                if target < lo:
                    target = lo + (lo - target)
                    if target > hi:
                        target = hi
                if target > hi:
                    overflow += mass
                    continue
                pos = (target - lo) / step
                base = int(math.floor(pos))
                frac = pos - base
                if base >= size - 1:
                    shifted[size - 1] += mass
                else:
                    shifted[base] += mass * (1.0 - frac)
                    shifted[base + 1] += mass * frac
            absent += overflow
            spread = [0.0] * size
            for j, mass in enumerate(shifted):
                if mass <= 0.0:
                    continue
                for off, w in kernels[j]:
                    jj = j + off
                    if 0 <= jj < size:
                        spread[jj] += mass * w
            exited = 0.0
            age_extra = 0.0
            if st["expiry"] is not None:
                remaining = (st["expiry"] - now) / 3600.0
                if remaining <= 0.0:
                    age_extra = AGE_POST_HAZARD
                elif remaining <= AGE_RAMP_HOURS:
                    age_extra = AGE_HAZARD * (1.0 - remaining / AGE_RAMP_HOURS)
            for j in range(size):
                mass = spread[j]
                if mass <= 0.0:
                    continue
                leave = min(0.9, exit_hazard[j] * EXIT_SCALE + age_extra)
                gone = mass * leave
                spread[j] = mass - gone
                exited += gone
            absent += exited
            returning = absent * min(0.9, ret_hazard * RETURN_SCALE)
            if returning > 0.0:
                absent -= returning
                for j in range(size):
                    spread[j] += returning * entry_weights[j]
            item[1] = spread
            item[2] = absent
            acc = totals[st["uid"]]
            for j, mass in enumerate(spread):
                place = place_of[j]
                if place > 0 and mass > 0.0:
                    acc[place] += mass / hours
    return totals


def assemble(totals, last_order, places):
    """Normalise per-place distributions, leaving the unnamed reserve, and
    order the top-10 guess by place-weighted mass."""
    probabilities = []
    strength = {}
    for place in range(1, places + 1):
        ranked = sorted(
            ((mass[place], uid) for uid, mass in totals.items() if mass[place] >= MIN_ROW_P),
            key=lambda item: (-item[0], item[1]),
        )[:TOPN_PER_PLACE]
        total = sum(mass for mass, _ in ranked)
        if total <= 0.0:
            continue
        scale = (1.0 - RESERVE) / total
        for mass, uid in ranked:
            p = mass * scale
            probabilities.append({"universe_id": uid, "place": place, "p": p})
            strength[uid] = strength.get(uid, 0.0) + p * (places + 1 - place)
    probabilities.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
    ordered = sorted(strength.items(), key=lambda item: (-item[1], item[0]))
    top10 = [uid for uid, _ in ordered[:places]]
    if len(top10) < places:
        for uid in last_order:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as handle:
            task = json.load(handle)
    except (OSError, ValueError):
        task = {}
    places = int(task.get("places") or 10)
    output_name = task.get("output") or "forecast.json"
    window = task.get("window") or {}
    try:
        hours = int(window.get("slots") or 72)
    except (TypeError, ValueError):
        hours = 72
    hours = min(max(hours, 1), 72 * 7)

    reads = read_history(input_dir)
    if not reads:
        forecast = {"probabilities": [], "top10": list(range(1, places + 1))}
    else:
        last_t = reads[-1]["t"]
        created = collect_created(reads)
        trust_age = bool(created) and age_signal_trustworthy(reads, created, last_t)
        centers, sigmas, hazards, ret_hazard = fit_rates(reads)
        grid = Grid(centers, sigmas)
        grid.set_hazards(centers, hazards)
        states = build_game_states(reads, centers, sigmas, hazards, ret_hazard,
                                   created, trust_age, last_t)
        totals = propagate(grid, states, hours, ret_hazard, last_t)
        forecast = assemble(totals, reads[-1]["order"], places)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(forecast, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
