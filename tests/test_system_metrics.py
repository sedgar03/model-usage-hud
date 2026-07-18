from __future__ import annotations

import unittest

from model_usage_hud.core import system_metrics as sm


# Real ``vm_stat`` output header + a representative subset of lines.
VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                3739.
Pages active:                            481165.
Pages inactive:                          445700.
Pages speculative:                        34198.
Pages wired down:                        222358.
Pages purgeable:                             26.
File-backed pages:                       310459.
Anonymous pages:                         650604.
Pages occupied by compressor:            853746.
"""


class VmStatParseTests(unittest.TestCase):
    def test_parses_page_size_and_counts(self) -> None:
        parsed = sm._parse_vm_stat(VM_STAT_SAMPLE)
        self.assertIsNotNone(parsed)
        page_size, counts = parsed
        self.assertEqual(page_size, 16384)
        self.assertEqual(counts["pages active"], 481165)
        self.assertEqual(counts["pages wired down"], 222358)
        self.assertEqual(counts["pages occupied by compressor"], 853746)

    def test_missing_page_size_returns_none(self) -> None:
        self.assertIsNone(sm._parse_vm_stat("no header here\nPages free: 10.\n"))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(sm._parse_vm_stat(""))


class SwapParseTests(unittest.TestCase):
    def test_parses_swapusage(self) -> None:
        # Feed the parser via the module-level regex the same way the code does.
        raw = "total = 2048.00M  used = 1245.88M  free = 802.12M  (encrypted)"
        match = sm._SWAP_RE.search(raw)
        self.assertIsNotNone(match)
        total = float(match.group(1)) * sm._UNIT_TO_BYTES[match.group(2)]
        used = float(match.group(3)) * sm._UNIT_TO_BYTES[match.group(4)]
        self.assertAlmostEqual(total / sm._BYTES_PER_GIB, 2.0, places=2)
        self.assertGreater(used, 0)


class PressureMappingTests(unittest.TestCase):
    def test_levels_map_to_labels(self) -> None:
        self.assertIsNone(sm._pressure_label(None))
        self.assertEqual(sm._pressure_label(1), "normal")
        self.assertEqual(sm._pressure_label(2), "warning")
        self.assertEqual(sm._pressure_label(4), "critical")
        # Unknown high value clamps to the most severe bucket.
        self.assertEqual(sm._pressure_label(8), "critical")


class BudgetAdviceTests(unittest.TestCase):
    def _snap(self, *, cpu_used, ncpu=10, available_gb, pressure, disk_free_gb, disk_used_pct):
        return {
            "cpu": {"used_pct": cpu_used, "ncpu": ncpu},
            "memory": {"available_gb": available_gb, "pressure": pressure},
            "disk": {"free_gb": disk_free_gb, "used_pct": disk_used_pct},
        }

    def test_healthy_machine_is_safe(self) -> None:
        budget = sm.derive_budget(
            self._snap(
                cpu_used=30, available_gb=40.0, pressure="normal",
                disk_free_gb=400.0, disk_used_pct=40.0,
            )
        )
        self.assertTrue(budget["advice"]["safe_to_start"])
        self.assertEqual(budget["cpu"]["free_cores"], 7.0)
        # 75% of available at normal pressure.
        self.assertEqual(budget["advice"]["suggested_mem_gb"], 30.0)

    def test_critical_pressure_blocks(self) -> None:
        budget = sm.derive_budget(
            self._snap(
                cpu_used=20, available_gb=2.0, pressure="critical",
                disk_free_gb=400.0, disk_used_pct=40.0,
            )
        )
        self.assertFalse(budget["advice"]["safe_to_start"])
        self.assertEqual(budget["advice"]["suggested_mem_gb"], 0.0)

    def test_cpu_pegged_blocks(self) -> None:
        budget = sm.derive_budget(
            self._snap(
                cpu_used=98, available_gb=20.0, pressure="normal",
                disk_free_gb=400.0, disk_used_pct=40.0,
            )
        )
        self.assertFalse(budget["advice"]["safe_to_start"])
        self.assertLess(budget["cpu"]["free_cores"], 1.0)

    def test_low_disk_blocks(self) -> None:
        budget = sm.derive_budget(
            self._snap(
                cpu_used=10, available_gb=20.0, pressure="normal",
                disk_free_gb=5.0, disk_used_pct=99.0,
            )
        )
        self.assertFalse(budget["advice"]["safe_to_start"])

    def test_missing_fields_do_not_raise(self) -> None:
        budget = sm.derive_budget({})
        self.assertIn("advice", budget)
        self.assertIsNone(budget["cpu"]["free_cores"])


class SnapshotShapeTests(unittest.TestCase):
    def test_collect_returns_expected_keys(self) -> None:
        snap = sm.collect_system_snapshot()
        for key in ("cpu", "memory", "swap", "disk", "platform"):
            self.assertIn(key, snap)
        # Disk works on any platform via shutil.
        self.assertIsNotNone(snap["disk"]["used_pct"])


if __name__ == "__main__":
    unittest.main()
