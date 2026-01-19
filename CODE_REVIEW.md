# Code Review - Package Implementation

**Date:** 2026-01-19  
**Reviewer:** AI Assistant  
**Status:** Documented - Fixes Pending

## Overview

This document contains a comprehensive code review of the channel generation package implementation. The review covers structure, performance, bugs, and design improvements.

## Structure & Organization

### ✅ **Status: Good**

The package has a well-organized structure with clear separation of concerns:

- **`src/scene_setup.py`** - Scene loading and preparation
- **`src/radio_map.py`** - Radio map solving and user sampling
- **`src/receivers.py`** - Receiver creation from sampled positions
- **`src/path_solver.py`** - Per-TX path solving with efficient removal/restoration
- **`src/channel.py`** - CFR computation and data saving
- **`src/channel_generator.py`** - Main API orchestrating all components

Each module has a single, focused responsibility, making the code maintainable and testable.

## Performance Issues

### 🔴 **Issue 1: Redundant `scene.get()` calls in `receivers.py`**

**Location:** `src/receivers.py`, Line 76

**Problem:**
```python
for tx_idx in range(num_txs):
    # ...
    tx_object = scene.get(tx_name)  # Called once per TX
    
    for user_idx in range(num_users_per_tx):
        # Uses tx_object - but this is fine
```

Actually, this is **NOT** a performance issue - `tx_object` is cached correctly outside the inner loop. The current implementation is correct.

**Status:** ✅ No issue - implementation is correct

---

### 🟡 **Issue 2: `.tolist()` calls in nested loops**

**Location:** `src/receivers.py`, Lines 82-83

**Problem:**
```python
for user_idx in range(num_users_per_tx):
    pos = positions[tx_idx, user_idx].tolist()  # Called for every user
    vel = velocities[user_count].tolist()        # Called for every user
```

**Impact:** Low - `.tolist()` is necessary for Sionna API compatibility. Pre-computing would require storing thousands of lists in memory.

**Recommendation:** Keep as-is. The current approach is reasonable.

**Status:** ✅ Acceptable - no change needed

---

### 🟡 **Issue 3: Missing `scene.scene_geometry_updated()` call**

**Location:** `src/path_solver.py`, After restoration in `finally` block

**Problem:**
After removing and restoring objects, Sionna might need to be notified that the scene geometry has changed.

**Current Code:**
```python
finally:
    # Restore removed receivers and TXs
    for rx_obj in removed_rxs:
        scene.add(rx_obj)
    for tx_obj in removed_txs:
        scene.add(tx_obj)
    # Missing: scene.scene_geometry_updated()?
```

**Impact:** Unknown - depends on Sionna's internal state management. May not be necessary if `scene.add()` handles it automatically.

**Recommendation:** Test if adding `scene.scene_geometry_updated()` (if method exists) improves reliability.

**Status:** 🟡 Low priority - needs testing

## Bugs Found

### 🔴 **Bug 1: Incorrect CFR combination when `per_tx_users_only=False`**

**Location:** `src/channel.py`, Line 74

**Problem:**
```python
# Get dimensions from first TX
num_selected_users, _, num_rx_ant, num_tx_ant, num_subcarriers_check, num_ofdm_symbols_check = cfr_per_tx[0].shape
num_txs = len(cfr_per_tx)
total_users = num_selected_users * num_txs  # ❌ BUG: Assumes per_tx_users_only=True
```

**Issue:**
- When `per_tx_users_only=True`: Each TX has `num_users_per_tx` users → `total_users = num_users_per_tx * num_txs` ✅ Correct
- When `per_tx_users_only=False`: All TXs use ALL users → `total_users = num_selected_users` (same for all TXs) ❌ Wrong calculation

**Impact:** 
- Array shape mismatch when `per_tx_users_only=False`
- Incorrect indexing when combining results

**Fix Required:**
1. Add `per_tx_users_only` parameter to `compute_cfr()`
2. Handle both cases:
   ```python
   if per_tx_users_only:
       total_users = num_selected_users * num_txs
   else:
       total_users = num_selected_users  # All TXs use same users
   ```

**Status:** 🔴 **CRITICAL** - Must fix before using `per_tx_users_only=False`

---

### 🟡 **Bug 2: Variable naming confusion in `channel_generator.py`**

**Location:** `src/channel_generator.py`, Line 148

**Problem:**
```python
# Calculate total number of TXs
num_txs = len(antenna_information) * num_sectors  # Calculated here

# ... later ...

num_txs_actual, num_users_per_tx, total_users = add_receivers_from_samples(...)  # Returns num_txs_actual
```

**Issue:**
- `num_txs` is calculated but never used
- `num_txs_actual` is used instead
- These should always match, but the naming is confusing

**Impact:** Low - doesn't cause bugs, but confusing for maintainers

**Recommendation:** Remove redundant `num_txs` calculation or use it consistently

**Status:** 🟡 Low priority - code clarity issue

## Design Improvements

### 🟡 **Improvement 1: Missing parameter in `compute_cfr()`**

**Location:** `src/channel.py`, Function signature

**Problem:**
`compute_cfr()` doesn't know whether `per_tx_users_only=True` or `False`, so it can't correctly combine results.

**Current Signature:**
```python
def compute_cfr(
    paths_per_tx: List,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    normalize_delays: bool = True,
    normalize: bool = True,
    out_type: str = "numpy"
) -> np.ndarray:
```

**Recommended:**
```python
def compute_cfr(
    paths_per_tx: List,
    num_subcarriers: int,
    num_ofdm_symbols: int,
    subcarrier_spacing: float,
    per_tx_users_only: bool = True,  # Add this parameter
    normalize_delays: bool = True,
    normalize: bool = True,
    out_type: str = "numpy"
) -> np.ndarray:
```

**Status:** 🟡 Medium priority - needed to fix Bug 1

---

### 🟡 **Improvement 2: Type hints could be improved**

**Location:** Multiple files

**Problem:**
Some functions use `Any` for scene objects:
```python
def setup_scene(...) -> Tuple[Any, dict, Optional[Any], list]:
```

**Recommendation:**
- Import proper types from `sionna.rt` if available
- Or create type aliases for better IDE support

**Status:** 🟡 Low priority - doesn't affect functionality

---

### 🟡 **Improvement 3: Error handling**

**Location:** Multiple files

**Current State:**
- Most functions don't have explicit error handling
- Relies on Python exceptions propagating up

**Recommendation:**
- Add validation for critical inputs (e.g., check if `mobility_preset` exists in `mobility_presets`)
- Add helpful error messages for common mistakes

**Status:** 🟡 Low priority - can be added incrementally

## Summary

### Critical Issues (Must Fix)
1. 🔴 **Bug 1:** CFR combination logic fails when `per_tx_users_only=False`

### Medium Priority (Should Fix)
1. 🟡 **Improvement 1:** Add `per_tx_users_only` parameter to `compute_cfr()`
2. 🟡 **Bug 2:** Clean up variable naming in `channel_generator.py`

### Low Priority (Nice to Have)
1. 🟡 **Issue 3:** Test `scene.scene_geometry_updated()` call
2. 🟡 **Improvement 2:** Improve type hints
3. 🟡 **Improvement 3:** Add error handling

### No Issues Found
- ✅ Structure and organization
- ✅ Performance of `.tolist()` calls (acceptable)
- ✅ Caching of `tx_object` (already correct)

## Recommendations

1. **Immediate Action:** Fix Bug 1 (CFR combination) before using `per_tx_users_only=False`
2. **Short-term:** Add `per_tx_users_only` parameter to `compute_cfr()` and update `channel_generator.py` to pass it
3. **Long-term:** Improve error handling and type hints incrementally

## Testing Notes

When fixing Bug 1, test with:
- `per_tx_users_only=True` (current default) - should work as before
- `per_tx_users_only=False` - should correctly combine all users for all TXs

Expected behavior when `per_tx_users_only=False`:
- Each TX's CFR has shape `[total_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]`
- Combined result should have shape `[total_users, num_txs, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]`
- All TXs use the same set of users (no slicing needed)
