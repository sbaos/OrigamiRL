"""
Test script to demonstrate the singularity fix near rho=π.

Shows that the original code produces discontinuous RBM results when crossing π,
while the fix produces continuous results where the two RBMs swap smoothly.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from ptu.ptu import calc_ptu, ptu

def test_singularity():
    print("=" * 70)
    print("Testing dihedral angles near π")
    print("=" * 70)
    
    PI = np.pi
    test_angles = np.linspace(3.10, 3.18, 9)
    
    print(f"\n{'rho':>8s} | {'RBM1_phi1':>10s} {'RBM1_phi2':>10s} {'RBM1_phi3':>10s} | {'RBM2_phi1':>10s} {'RBM2_phi2':>10s} {'RBM2_phi3':>10s}")
    print("-" * 85)
    
    prev_M1 = None
    
    for rho in test_angles:
        sector_angles = [[rho/2, rho/2], [rho/2], [rho/2]]
        list_in_diheral_angles = [[rho], [], []]
        
        t, M1, M2 = calc_ptu(sector_angles, list_in_diheral_angles)
        
        if M1 and M2:
            # Check if RBMs swapped (compare with previous step)
            swapped = ""
            if prev_M1 is not None:
                d1 = sum(abs(a - b) for a, b in zip(M1, prev_M1))
                d2 = sum(abs(a - b) for a, b in zip(M2, prev_M1))
                if d2 < d1:
                    swapped = " ← SWAP detected!"
            
            print(f"{rho:8.4f} | {M1[0]:10.4f} {M1[1]:10.4f} {M1[2]:10.4f} | {M2[0]:10.4f} {M2[1]:10.4f} {M2[2]:10.4f}{swapped}")
            prev_M1 = M1
        else:
            print(f"{rho:8.4f} | {'FAILED':>10s}")

    print("\n" + "=" * 70)
    print("Solution: Track continuity by selecting the RBM closer to previous step")
    print("=" * 70)
    
    prev_M1 = None
    prev_M2 = None
    
    print(f"\n{'rho':>8s} | {'Cont_M1_phi1':>12s} {'Cont_M1_phi2':>12s} {'Cont_M1_phi3':>12s} | {'Cont_M2_phi1':>12s} {'Cont_M2_phi2':>12s} {'Cont_M2_phi3':>12s}")
    print("-" * 95)
    
    for rho in test_angles:
        sector_angles = [[rho/2, rho/2], [rho/2], [rho/2]]
        list_in_diheral_angles = [[rho], [], []]
        
        t, M1, M2 = calc_ptu(sector_angles, list_in_diheral_angles)
        
        if M1 and M2:
            # Ensure continuity
            if prev_M1 is not None:
                d_M1_to_prev1 = sum(abs(a - b) for a, b in zip(M1, prev_M1))
                d_M2_to_prev1 = sum(abs(a - b) for a, b in zip(M2, prev_M1))
                if d_M2_to_prev1 < d_M1_to_prev1:
                    M1, M2 = M2, M1  # swap to maintain continuity
            
            print(f"{rho:8.4f} | {M1[0]:12.4f} {M1[1]:12.4f} {M1[2]:12.4f} | {M2[0]:12.4f} {M2[1]:12.4f} {M2[2]:12.4f}")
            prev_M1 = M1
            prev_M2 = M2
        else:
            print(f"{rho:8.4f} | {'FAILED':>12s}")

if __name__ == "__main__":
    test_singularity()
