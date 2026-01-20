#tests for spectral operations (cleaning/matching)
import pytest
import numpy as np
from TunaSimNetwork.datasetBuilder import specCleaner, trainSetBuilder

class TestMatching:

    def test_0(self):
        """ 
        empty specs
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([])

        t = np.array([])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 0.5,
                            units_ppm = False
                            )

        assert len(matched_a) == 0 and len(matched_b) == 0
    
    def test_1(self):
        """ 
        all match with no overlap
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,10],
             [11, 11],
             [12,12]])

        t = np.array([[10,10],
             [11, 11],
             [12,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 0.5,
                            units_ppm = False
                            )

        answer_a, answer_b = (np.array([10,11,12]), np.array([10,11,12]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_2(self):
        """ 
        mix of match and non match, zero indices should be removed
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,0],
             [11, 11],
             [12,12]])

        t = np.array([[10,10],
             [11, 11],
             [13,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 0.5,
                            units_ppm = False
                            )

        answer_a, answer_b = (np.array([11, 12, 0, 0]), np.array([11, 0, 10, 12]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_3(self):
        """ 
        bridging peak 
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,10],
             [11, 11],
             [11.5,12]])

        t = np.array([[10.6,10],
             [11.1, 11],
             [12,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 0.55,
                            units_ppm = False
                            )

        answer_a, answer_b = (np.array([10, 11, 12, 0]), np.array([0, 11, 12, 10]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_0_ppm(self):
        """ 
        empty specs
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([])

        t = np.array([])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 1e6,
                            units_ppm = True
                            )

        answer_a, answer_b = (np.array([]), np.array([]))

        assert len(answer_a) == 0 and len(answer_b) == 0

    def test_1_ppm(self):
        """ 
        all match with no overlap
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,10],
             [11, 11],
             [12,12]])

        t = np.array([[10,10],
             [11, 11],
             [12,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 5e4,
                            units_ppm = False
                            )

        answer_a, answer_b = (np.array([10,11,12]), np.array([10,11,12]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_2_ppm(self):
        """ 
        mix of match and non match, zero indices should be removed
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,0],
             [11, 11],
             [12,12]])

        t = np.array([[10,10],
             [11, 11],
             [13,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 5e4,
                            units_ppm = True
                            )

        answer_a, answer_b = (np.array([11, 12, 0, 0]), np.array([11, 0, 10, 12]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_3_ppm(self):
        """ 
        bridging peak 
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,10],
             [11, 11],
             [11.5,12]])

        t = np.array([[10.6,10],
             [11.1, 11],
             [12,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 5.5e4,
                            units_ppm = True
                            )

        answer_a, answer_b = (np.array([10, 11, 12, 0]), np.array([0, 11, 12, 10]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)

    def test_4_ppm(self):
        """ 
        one included, one excluded based on ppm
        """

        matcher = trainSetBuilder('',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  3,
                                  )
        
        q = np.array([[10,10],
             [11, 11],
             [12,12]])

        t = np.array([[10.11,10],
             [11, 11],
             [12.11,12]])

        matched_a, matched_b = matcher.match_spectra(q,
                            t,
                            tolerance = 1e4,
                            units_ppm = True
                            )

        answer_a, answer_b = (np.array([10, 11, 12, 0,]), np.array([0, 11, 12, 10]))

        assert np.all(matched_a == answer_a) and np.all(matched_b == answer_b)
      

class TestDeisotoping:

    def test_0(self):

        cleaner = specCleaner()

        input = np.array([[]], dtype = np.float64)

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)

        assert output.size == 0

    
    def test_1(self):
        """
        one isotope to consolidate, first peak is monoiso
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [60., 60],
                          [61, 61],
                          [62.0033, 62],
                          [68, 68]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 50],
                          [60., 60],
                          [62.0033, 123],
                          [68, 68]])

        assert np.all(output == answer)

    def test_2(self):
        """
        no isotopic peaks exist
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [59., 60],
                          [61, 61],
                          [62.04, 62],
                          [68, 68]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 50],
                          [59., 60],
                          [61, 61],
                          [62.04, 62],
                          [68, 68]])

        assert np.all(output == answer)

    def test_3(self):
        """
        two different fragements with isotopic peaks, first has lower mz monoiso, 2nd does not
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [51.00299, 40],
                          [60., 60],
                          [61, 61],
                          [62.0033, 62],
                          [68, 68]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 90],
                          [60., 60],
                          [62.0033, 123],
                          [68, 68]])

        assert np.all(output == answer)

    def test_4(self):
        """
        chained isotopic peaks, middle is monoisotope
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [51.003355, 40],
                          [60., 60],
                          [61, 61],
                          [62.003, 62],
                          [63.006, 40]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 90],
                          [60., 60],
                          [62.003, 163]])

        assert np.all(output == answer)

    def test_5(self):
        """
        chained isotopic peaks, middle is monoisotope. Other peaks around
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [51.003355, 40],
                          [60., 60],
                          [61, 61],
                          [62.003, 62],
                          [62.5, 65],
                          [63.006, 40]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 90],
                          [60., 60],
                          [62.003, 163],
                          [62.5, 65],])

        assert np.all(output == answer)

    def test_6(self):
        """
        chained isotopic peaks, middle is monoisotope. Other peaks around. additional at
        """

        cleaner = specCleaner()

        input = np.array([[50., 50],
                          [51.003355, 40],
                          [60., 60],
                          [61, 61],
                          [62.003, 62],
                          [62.5, 65],
                          [63.006, 40],
                          [64, 64]])

        output = cleaner.consolidate_isotopic_peaks(input,
                                                    1.003355,
                                                    0.001)
        
        print(output)

        answer = np.array([[50., 90],
                          [60., 60],
                          [62.003, 163],
                          [62.5, 65],
                          [64, 64]])

        assert np.all(output == answer)
