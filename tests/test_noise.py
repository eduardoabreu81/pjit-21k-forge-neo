# P-JiT 21K support for Forge Neo
# The port of upstream's seeded noise: Philox against Random123's known answers, torch's thread layout against upstream's
# formula, and the normals' statistics. Upstream's own kernel needs a CUDA GPU; the pictures Linum published are the check
# of the whole, on the GPU.

import math
import unittest

import numpy as np
import torch

from forge_pjit.native import noise


def words(*values):
    return [np.array([v], dtype=np.uint64) for v in values]


class TestPhilox(unittest.TestCase):
    def check(self, key, counter, expected):
        got = noise.philox(key[0], key[1], *words(*counter))
        self.assertEqual([int(w[0]) for w in got], expected)

    # Random123's kat_vectors for philox4x32 with 10 rounds
    def test_zeros(self):
        self.check((0, 0), (0, 0, 0, 0), [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8])

    def test_ones(self):
        ones = 0xFFFFFFFF
        self.check((ones, ones), (ones, ones, ones, ones), [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD])

    def test_digits_of_pi(self):
        self.check((0xA4093822, 0x299F31D0), (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
                   [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1])


class TestRandn(unittest.TestCase):
    def test_shape_and_statistics(self):
        x = noise.randn(2, (3, 1216, 864))
        self.assertEqual(x.shape, (3, 1216, 864))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLess(abs(x.mean().item()), 0.01)
        self.assertLess(abs(x.std().item() - 1.0), 0.01)
        self.assertFalse(torch.isnan(x).any())

    def test_seeds(self):
        self.assertTrue(torch.equal(noise.randn(5, (3, 64, 64)), noise.randn(5, (3, 64, 64))))
        self.assertFalse(torch.equal(noise.randn(5, (3, 64, 64)), noise.randn(6, (3, 64, 64))))

    def test_thread_layout(self):
        # a small tensor has as many threads as blocks fit it: element e comes from stream e % T, word (e // T) % 4
        numel = 1000
        threads = noise.BLOCK * math.ceil(numel / noise.BLOCK)
        x = noise.randn(9, (numel,)).numpy()
        for element in (0, 1, 255, 256, 999):
            stream = np.array([element % threads], dtype=np.uint64)
            r0, r1, r2, r3 = noise.philox(9, 0, *words(0, 0), stream, *words(0))
            normals = (*noise.box_muller(r0, r1), *noise.box_muller(r2, r3))
            self.assertEqual(x[element], normals[(element // threads) % 4][0])

    def test_the_sm_count_changes_the_draw_only_when_it_binds(self):
        small = (3, 64, 64)  # below the SM cap
        self.assertTrue(torch.equal(noise.randn(1, small, sm_count=132), noise.randn(1, small, sm_count=114)))
        large = (3, 1024, 1024)
        self.assertFalse(torch.equal(noise.randn(1, large, sm_count=132), noise.randn(1, large, sm_count=114)))


if __name__ == "__main__":
    unittest.main()
