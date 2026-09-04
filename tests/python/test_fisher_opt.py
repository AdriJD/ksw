import unittest
import numpy as np

from ksw import fisher_opt

class TestFisherOpt(unittest.TestCase):
    
    def setUp(self):
        # Is called before each test.
        pass
    
    def tearDown(self):
        # Is called after each test.
        pass

    def test_fisher_opt(self):


        def _fisher_distance_bruteforce(F, indices, rcond=1e-10):
            ''' Eq. 37 from Smith & Zaldarriaga 2011. '''
            Nfact = F.shape[0]
            mask = np.zeros(Nfact, dtype=bool)
            mask[indices] = True
            F00 = F[np.ix_(mask, mask)]
            F01 = F[np.ix_(mask, ~mask)]
            F11 = F[np.ix_(~mask, ~mask)]
            F00_pinv = np.linalg.pinv(F00, rcond=rcond)
            resid = F11 - F01.T @ F00_pinv @ F01
            return float(np.sum(resid))

        def _optimal_weights_bruteforce(F, indices):
            ''' Eq. 36 from Smith & Zaldarriaga 2011. '''            
            indices = np.asarray(indices)
            nfact = F.shape[0]
            not_indices = np.setdiff1d(np.arange(nfact), indices)
            F00 = F[np.ix_(indices, indices)] # Preserve order of `indices`.
            F01 = F[np.ix_(indices, not_indices)]
            # First do the sum over J (F01 @ ones), then multiply by F00^-1.
            ones = np.ones(F01.shape[1])
            return 1.0 + np.linalg.solve(F00, F01 @ ones)               

        rng = np.random.default_rng(0)

        n_basis = 5 # Number of "true" independent directions.
        nfact = 400 # Oversampled number of terms.
        
        # Random combination coefficients mapping n_basis "true" directions
        # into nfact redundant terms.
        A = rng.normal(size=(nfact, n_basis))

        # Positive semi-definite Fisher matrix.
        M = rng.normal(size=(n_basis, n_basis))
        M = M @ M.T + n_basis * np.eye(n_basis)
        F = A @ M @ A.T

        threshold = 1e-3        
        indices, weights, score = fisher_opt.optimize_bispectrum(
            F, threshold=threshold, verbose=False)

        score_ref = _fisher_distance_bruteforce(F, indices)
        weights_ref = _optimal_weights_bruteforce(F, indices)

        self.assertTrue(score < threshold)
        self.assertTrue(len(indices) == n_basis - 1) # Note, this is expected given low threshold.
        self.assertTrue(abs(score - score_ref) < threshold)
        np.testing.assert_allclose(score, score_ref, rtol=1e-3)
        np.testing.assert_allclose(weights, weights_ref)
        
        F00 = F[np.ix_(indices, indices)]
        np.testing.assert_allclose(float(weights @ F00 @ weights), np.sum(F), rtol=1e-8)
        
        # Again but with more stringent threshold. Now the final score is numerically
        # instable, but we should recover all terms.
        threshold = 1e-10
        indices, weights, score = fisher_opt.optimize_bispectrum(
            F, threshold=threshold, verbose=False)

        score_ref = _fisher_distance_bruteforce(F, indices)
        weights_ref = _optimal_weights_bruteforce(F, indices)

        self.assertTrue(score < threshold) # score likely negative.
        self.assertTrue(len(indices) == n_basis)
        # Only test absolute, relative different instable.
        self.assertTrue(abs(score) - abs(score_ref) < 1e-8)
        np.testing.assert_allclose(weights, weights_ref)

        F00 = F[np.ix_(indices, indices)]
        np.testing.assert_allclose(float(weights @ F00 @ weights), np.sum(F), rtol=1e-12)
