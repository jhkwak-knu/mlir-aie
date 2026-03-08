# Fix: PRES Packet Flow Routing Failure in AIE2 Switchbox

## Table of Contents

- [Problem Statement](#problem-statement)
- [Background: AIE2 Switchbox Architecture](#background-aie2-switchbox-architecture)
- [Root Cause Analysis](#root-cause-analysis)
  - [Issue 1: Packet ID False Match at Intermediate Switchboxes](#issue-1-packet-id-false-match-at-intermediate-switchboxes)
  - [Issue 2: AMSEL Assignment Skipping Exact Matches](#issue-2-amsel-assignment-skipping-exact-matches)
  - [Issue 3: Flow Processing Order Dependency](#issue-3-flow-processing-order-dependency)
- [Solution](#solution)
  - [Fix 1: Destination-Based Bitmask Packet IDs](#fix-1-destination-based-bitmask-packet-ids)
  - [Fix 2: Prefer Exact Match in AMSEL Assignment](#fix-2-prefer-exact-match-in-amsel-assignment)
  - [Fix 3: Sort Packet Flows by Destination Count](#fix-3-sort-packet-flows-by-destination-count)
- [Verification Results](#verification-results)
- [Commits](#commits)
- [Appendix: Detailed Walkthrough](#appendix-detailed-walkthrough)

---

## Problem Statement

When the ONNXToAIE pass generates MLIR for configurations that require
**partial-sum forwarding** (`needsPres()` — i.e., `TPk > 1 && tpOrder[0] != K`),
the downstream pathfinder routing pass fails with:

```
error: 'aie.masterset' op a master port can only be tied to one arbiter
```

This error occurs during `AIECreatePathFindFlows`, which translates logical
packet flows into physical switchbox routing (arbiter/msel assignments).

**Affected configurations** (all failed prior to fix):

| SP layout | tpOrder | PRES needed | Result |
|-----------|---------|-------------|--------|
| (2,2)     | [0,2,1] | Yes (M-axis)| FAIL   |
| (2,2)     | [1,2,0] | Yes (N-axis)| FAIL   |
| (1,4)     | [0,2,1] | Yes (M-axis)| FAIL   |

Non-PRES configurations (`tpOrder[0] == K`) were not affected and always passed.

---

## Background: AIE2 Switchbox Architecture

Understanding the problem requires knowledge of three AIE2 switchbox concepts:

### 1. Packet ID Routing with Mask/Value Pairs

Each switchbox tile routes packets using 5-bit **mask/value** pairs on its
slave ports. A packet with header ID `H` matches a rule if:

```
(H & mask) == value
```

For example, `mask=0b00010, value=0b00010` matches any packet whose bit 1 is
set (IDs 2, 3, 6, 7, 10, ...). The pathfinder automatically derives mask/value
pairs from the set of packet IDs that must be routed through each port.

### 2. Arbiter and Msel Resources

Each master port (output) in a switchbox tile is assigned to exactly **one
arbiter**. Each tile has **6 arbiters**, and each arbiter has **4 msels**
(match-select slots). A packet flow is routed by assigning it an `amsel`
(arbiter + msel combination). Multiple flows sharing the same master port
**must** share the same arbiter — violating this produces the error above.

### 3. Multicast Flows

A single packet flow can target multiple destination ports (multicast). For
example, an LHS data packet may need to be delivered to two compute tiles in
the same column. At intermediate switchbox tiles, the packet must be
**forwarded** to the next tile while also being **delivered** locally, creating
a compound destination set like `{DMA:0, North:4}`.

---

## Root Cause Analysis

Three independent issues combined to cause the routing failure. All three had
to be fixed simultaneously — fixing any one or two alone was insufficient.

### Issue 1: Packet ID False Match at Intermediate Switchboxes

**File**: `lib/Conversion/ONNXToAIE/AiePlacement.cpp`

#### Previous Packet ID Assignment (Sequential Power-of-2)

```cpp
pkt.packetId = (1u << lhsPacketCnt++);  // LHS: 1, 2, 4, 8
pkt.packetId = (1u << rhsPacketCnt++);  // RHS: 1, 2, 4, 8
pkt.packetId = (1u << presPacketCnt++) + PRES_PKT_ID_OFFSET;  // PRES: 17, 18, 20, 24
```

The IDs encoded a **sequential counter** (which flow within its type), not
**where the flow goes**. This was fine when flows had simple 1-to-1 routing,
but caused collisions with multicast + PRES flows.

#### Collision Scenario

Consider column 0 with SP=(2,2):

```
row 3: comp_tile (0,3)  — position 1 in column
row 2: comp_tile (0,2)  — position 0 in column
row 1: mem_tile
row 0: shim_tile
```

LHS0 is a multicast flow: shim → comp(0,2) **and** comp(0,3) (same `m_idx`).
At tile (0,3), the switchbox must:
- **Deliver** LHS0 to local DMA (this tile needs the data)
- **Forward** LHS0 southward to comp(0,2)

PRES1 is a unicast flow: comp(0,3) DMA → shim (southward).

Both LHS0 and PRES1 pass through the South master port of tile (0,3).
The pathfinder must create mask/value rules to distinguish them. With
sequential IDs:

```
LHS0 ID  = 1  = 0b00001
PRES1 ID = 18 = 0b10010
```

The pathfinder looks for bit positions that differentiate these IDs, but with
multiple flows competing for the same 5-bit space, it can produce identical
mask/value pairs for different flows — a **false match** where the hardware
cannot distinguish which packets belong to which flow.

### Issue 2: AMSEL Assignment Skipping Exact Matches

**File**: `lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp` (line 536-543)

When assigning an amsel to a new packet flow, the algorithm searches existing
amsels to find one that can be **reused** (avoiding unnecessary msel consumption):

- **Exact match**: The existing amsel's destination port set is identical to the
  new flow's destinations. The amsel can be fully reused — zero additional msel
  cost.
- **Partial match**: The existing amsel's destinations overlap with (but are not
  identical to) the new flow's. The same arbiter can be used, but a new msel
  must be allocated.

#### The Bug

```cpp
// BEFORE (buggy):
if (matched) {
    foundMatchedDest = true;
    if (mismatched)
        foundPartialMatchArbiter = getArbiterIDFromAmsel(amselValue);
    else if (ports.size() != packetFlow.second.size())
        foundPartialMatchArbiter = getArbiterIDFromAmsel(amselValue);
    break;  // ← BUG: exits on first match, even if it's only partial
}
```

The `break` statement exits the search loop on the **first** match — even if
it is only a partial match. An exact match later in the map is never discovered.
This causes each new flow to consume a fresh msel where reuse was possible,
eventually exhausting the 4-msel-per-arbiter limit.

### Issue 3: Flow Processing Order Dependency

**File**: `lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp` (line 479)

The AMSEL algorithm processes flows in map iteration order, which is
non-deterministic with respect to destination count. The processing order
determines which flow "establishes" an arbiter for a shared master port.

#### Order-Dependent Failure

At tile (0,3), two flows share master port DMA:0:

```
Flow A (PRES):  destinations = {DMA:0}            — 1 port
Flow B (LHS):   destinations = {DMA:0, North:4}   — 2 ports
```

**If PRES is processed first:**

1. PRES `{DMA:0}` → creates arbiter 5, assigns msel 0
2. LHS `{DMA:0, North:4}` → finds DMA:0 has arbiter 5 (partial match)
   → North:4 has no arbiter yet → must create arbiter 4 for North:4
   → But DMA:0 is now associated with **both** arbiter 5 (from PRES)
     and arbiter 4 (from LHS)
   → **ERROR: "master port tied to two arbiters"**

**If LHS is processed first:**

1. LHS `{DMA:0, North:4}` → creates arbiter 4, assigns msel 0
2. PRES `{DMA:0}` → finds DMA:0 has arbiter 4 (partial match)
   → reuses arbiter 4, assigns msel 1
   → DMA:0 uses only arbiter 4
   → **Success**

---

## Solution

### Fix 1: Destination-Based Bitmask Packet IDs

**File**: `lib/Conversion/ONNXToAIE/AiePlacement.cpp`

Replace sequential counter IDs with IDs that encode the **destination tile
position** within the column as a bitmask:

```cpp
// BEFORE (sequential):
pkt.packetId = (1u << lhsPacketCnt++);

// AFTER (destination-based):
pkt.packetId = (1u << i);               // i = tile position in column
// For multicast, OR the bits of all destination tiles:
it->second.packetId |= (1u << i);
```

**Example for SP=(2,2), column 0:**

| Flow | Targets | ID (binary) | ID (decimal) |
|------|---------|-------------|--------------|
| LHS0 (multicast) | comp(0,2) + comp(0,3) | `0b00011` | 3 |
| RHS0 | comp(0,2) only | `0b00001` | 1 |
| RHS1 | comp(0,3) only | `0b00010` | 2 |

At intermediate tile (0,3), bit 1 distinguishes "deliver here" (bit 1 set)
from "pass through for comp(0,2)" (bit 1 clear). The pathfinder's mask/value
derivation can always find a distinguishing bit.

**PRES ID separation**: PRES flows share the DMA input channel with LHS, so
they add `PRES_PKT_ID_OFFSET = 16` (bit 4) to guarantee type separation:

```
LHS0  ID = 3  = 0b00011
PRES0 ID = 17 = 0b10001
→ bit 4 immediately distinguishes LHS from PRES
```

**Additional fix (pres buffer resolution)**: On compute tiles, PRES data is
received into the `"res"` buffer (the accumulator), not a separate `"pres"`
buffer. Only shim tiles use the dedicated `"pres"` buffer:

```cpp
if (packet.name.compare(0, 4, "pres") == 0) {
    bufName = (dstTile.row == device.shimRow) ? "pres" : "res";
} else {
    bufName = comm.name;
}
```

### Fix 2: Prefer Exact Match in AMSEL Assignment

**File**: `lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp`

Only `break` on exact match; for partial matches, record the arbiter but
continue searching:

```cpp
// AFTER (fixed):
if (matched) {
    foundMatchedDest = true;
    if (!mismatched && ports.size() == packetFlow.second.size()) {
        // Exact match — reuse this amsel entirely, no new msel needed.
        foundPartialMatchArbiter = -1;
        break;
    }
    // Partial match — record the arbiter but keep searching.
    // An exact match later in the map can save an msel.
    if (foundPartialMatchArbiter < 0)
        foundPartialMatchArbiter = getArbiterIDFromAmsel(amselValue);
}
```

This ensures that an exact match is always found when it exists, minimising
msel consumption and preventing arbiter exhaustion.

### Fix 3: Sort Packet Flows by Destination Count

**File**: `lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp`

Sort flows so that those with more destinations are processed first:

```cpp
std::stable_sort(sortedPacketFlows.begin(), sortedPacketFlows.end(),
                 [](const auto &a, const auto &b) {
                   return a.second.size() > b.second.size();
                 });
```

Multi-destination flows (e.g., LHS multicast `{DMA:0, North:4}`) establish the
arbiter for all their ports first. Single-destination flows (e.g., PRES
`{DMA:0}`) processed later find a partial match on the same arbiter and reuse
it — no conflicting arbiter is ever created.

---

## Verification Results

All three previously-failing PRES configurations now pass, along with
non-PRES regression:

| Test Case | tpOrder | PRES | SP | Result | NPU Time |
|-----------|---------|------|----|--------|----------|
| M-axis pres | [0,2,1] | Yes | (2,2) | **PASS** | 141.96 us |
| N-axis pres | [1,2,0] | Yes | (2,2) | **PASS** | 118.33 us |
| Non-pres regression | [2,0,1] | No | (2,2) | **PASS** | 112.29 us |

Test configuration: M=32, K=32, N=32, elemType=bf16, numCores=4, TPk=2.

---

## Commits

| Commit | Branch | Description |
|--------|--------|-------------|
| `e6cc76b7` | `fix/pathfinder-amsel-assignment` (from `main`) | Upstream bugfix: AMSEL exact-match preference + flow sorting |
| `fbd35ba5` | `spatio-temporal-cost-model` | Destination-based packet IDs + pres buffer resolution |

The upstream bugfix was committed on a separate branch from `main` and merged
into `spatio-temporal-cost-model`, as it addresses a general pathfinder issue
independent of the ONNXToAIE pass.

---

## Appendix: Detailed Walkthrough

### A1. Switchbox Routing at Tile (0,3) — Before vs. After

**Before fix (sequential IDs, PRES processed first):**

```
tile(0,3) switchbox:

Slave ports                  Master ports
  South:3 ──┐                ┌── DMA:0    (arbiter 5, msel 0 = PRES)
  DMA:0  ──┐│                │            (arbiter 4, msel 0 = LHS) ← CONFLICT!
            ││   ┌────────┐  │
            └┼──►│ packet │──┤
             └──►│ switch │──┼── North:4  (arbiter 4, msel 0 = LHS)
                 └────────┘  └── South:3  (arbiter 5, msel 0 = PRES)

ERROR: DMA:0 cannot be tied to both arbiter 4 and arbiter 5.
```

**After fix (destination-based IDs, LHS processed first):**

```
tile(0,3) switchbox:

Slave ports                  Master ports
  South:3 ──┐                ┌── DMA:0    (arbiter 4, msel 0 = LHS, msel 1 = PRES)
  DMA:0  ──┐│                │
            ││   ┌────────┐  │
            └┼──►│ packet │──┤
             └──►│ switch │──┼── North:4  (arbiter 4, msel 0 = LHS)
                 └────────┘  └── South:3  (arbiter 4, msel 1 = PRES)

OK: DMA:0 uses only arbiter 4. LHS and PRES share the arbiter with
    different msels and are distinguished by packet ID mask/value.
```

### A2. Packet ID Comparison Table

**Sequential (before):**

| Flow | ID | Binary | Problem |
|------|----|--------|---------|
| LHS0 | 1 | `00001` | At tile (0,3), only bit 0 differs from PRES |
| LHS1 | 2 | `00010` | No spatial meaning |
| PRES0 | 17 | `10001` | bit 0 same as LHS0 — potential mask/value collision |
| PRES1 | 18 | `10010` | bit 1 same as LHS1 |

**Destination-based (after):**

| Flow | Targets | ID | Binary | Distinguishing bits |
|------|---------|----|--------|---------------------|
| LHS0 (multicast) | tile 0+1 | 3 | `00011` | bit 0,1 encode destinations |
| RHS0 | tile 0 | 1 | `00001` | bit 0 = tile 0 |
| RHS1 | tile 1 | 2 | `00010` | bit 1 = tile 1 |
| PRES0 | tile 0 | 17 | `10001` | bit 4 separates from LHS |
| PRES1 | tile 1 | 18 | `10010` | bit 4 separates from LHS |

Each tile position has a unique bit, so the pathfinder can always derive a
mask/value pair that correctly routes packets at every intermediate switchbox.
