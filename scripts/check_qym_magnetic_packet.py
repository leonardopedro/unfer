"""Independent continuous integrals for the unreduced SU(2) magnetic packet.

Run with python3 scripts/check_qym_magnetic_packet.py from any directory.
The integration nodes evaluate packet integrals; they do not define a spatial lattice.
"""
from fractions import Fraction
import json
from math import cos, exp, fsum, isqrt, pi, sin
from pathlib import Path


def integrate(function, radius, panels):
    """Composite Simpson rule on a symmetric interval (even panel count)."""
    step = 2 * radius / panels
    total = fsum((4 if i % 2 else 2) * function(-radius + i * step)
                 for i in range(1, panels))
    return step * (function(-radius) + total + function(radius)) / 3


def coordinate(panels):
    def profile(x):
        return exp(-pi * x*x)

    def derivative_pair(x):
        u = v = profile(x)
        w = x * profile(x)
        dv = -2*pi*x*v
        dw = profile(x) - 2*pi*x*w
        return u * (v*dw - w*dv)

    along_z = integrate(lambda z: profile(z)**3 * cos(2*pi*z) *
                        cos(4*pi*z) * cos(6*pi*z), 6, panels)
    return integrate(derivative_pair, 6, panels) * integrate(
        lambda y: profile(y)**3, 6, panels) * along_z


def momentum(panels):
    def bell(p):
        return exp(-pi*p*p)

    def packet(p, centre):
        return (bell(p-centre) + bell(p+centre)) / 2

    def pair(test, left, right):
        def outer(k):
            return test(k) * integrate(lambda q: left(q)*right(k-q), 12, panels)
        return integrate(outer, 12, panels)

    # Factors of i are multiplied after conjugating the differentiated test leg.
    first_x = pair(lambda k: 2*pi*k*k*bell(k), bell, bell)
    second_x = pair(lambda k: 2*pi*k*bell(k), bell, lambda p: p*bell(p))
    common_y = pair(bell, bell, bell)
    def z_channel(a, b, c):
        return pair(lambda k: packet(k, c), lambda q: packet(q, a), lambda p: packet(p, b))
    return (first_x * common_y * z_channel(1, 2, 3),
            second_x * common_y * z_channel(1, 3, 2))


def check_action_coefficients(coefficients):
    """Integrate the profile metric, all curls and both nonzero quartic terms in x."""
    def integral(f):
        return integrate(f, 6, 768)

    def base(x):
        return exp(-pi*x*x)

    def longitudinal(z, j):
        return base(z)*cos(2*pi*j*z)

    for index, j in enumerate((1, 2, 3)):
        def profile(x):
            return (x if j == 3 else 1)*base(x)

        def derivative(x):
            return ((1-2*pi*x*x) if j == 3 else -2*pi*x)*base(x)

        def dz(z):
            return -2*pi*base(z)*(z*cos(2*pi*j*z) + j*sin(2*pi*j*z))

        xx = integral(lambda x: profile(x)**2)
        yy = integral(lambda y: base(y)**2)
        zz = integral(lambda z: longitudinal(z,j)**2)
        mass = xx*yy*zz
        curl = yy*(integral(lambda x: derivative(x)**2)*zz + xx*integral(lambda z: dz(z)**2))
        assert abs(mass-coefficients['profile_metric'][index]) < 1e-13
        assert abs(curl-coefficients['curl_norm_squared'][index]) < 1e-12
        # The squares involving direction 3 are purely linear on this three-profile ansatz.
        assert integral(lambda z: dz(z)**2) >= pi*zz
        if j > 1:
            wall = (integral(lambda x: (base(x)*profile(x))**2) * integral(lambda y: base(y)**4)
                    * integral(lambda z: (longitudinal(z,1)*longitudinal(z,j))**2))
            assert abs(wall-coefficients['wall_12_13'][j-2]) < 1e-13


def exp_negative_bounds(x):
    """For 0 ≤ x/64 < 1 bracket exp(-x/64) by alternating partial sums."""
    y = x / 64
    assert 0 <= y < 1
    term = total = Fraction(1)
    for degree in range(1, 17):
        term *= -y / degree
        total += term
    # Degrees 15 and 16. Coarsen outwards before raising to the 64th power.
    scale = 10**19
    lo = (total-term)*scale
    hi = total*scale
    return (Fraction(lo.__floor__(), scale)**64, Fraction(hi.__ceil__(), scale)**64)


def exact_enclosure():
    """No floating point in the enclosure of the analytically integrated expression."""
    scale = 10**20
    # Real.pi_gt_d20 and Real.pi_lt_d20 (Mathlib).
    pi_lo = Fraction(314159265358979323846, scale)
    pi_hi = pi_lo + Fraction(1, scale)
    sqrt_lo = Fraction(isqrt(3*scale**2), scale)
    sqrt_hi = sqrt_lo + Fraction(1, scale)
    assert sqrt_lo**2 <= 3 <= sqrt_hi**2
    lower = upper = Fraction()
    for n in (0, 2, 4, 6):
        lower += exp_negative_bounds(pi_hi*n*n/3)[0]
        upper += exp_negative_bounds(pi_lo*n*n/3)[1]
    return lower / (12*sqrt_hi), upper / (12*sqrt_lo)


def main():
    reference = Path(__file__).resolve().parents[1] / 'docs/fixtures/qym_magnetic_packet.json'
    record = json.loads(reference.read_text())
    assert record['schema'] == 'qym-cubic-continuous-packet/v1' and record['coupling'] == 1
    check_action_coefficients(record['packet_action'])
    lo, hi = exact_enclosure()
    target = Fraction(record['entry_decimal'])
    assert lo <= target <= hi and hi-lo < Fraction(1, 10**15)
    for panels in (384, 768):
        direct = coordinate(panels)
        colour_three, colour_two = momentum(panels)
        for observed, expected in ((direct, target), (colour_three, 2*target/3),
                                   (colour_two, target/3)):
            assert abs(observed - float(expected)) < 2e-12, (panels, observed, float(expected))
        print(f'{panels} panels: x={direct:.15f}, k={colour_three+colour_two:.15f}')
    # A real paired cosine is even: test the transfer sign on a single polarized component.
    def polarized(q, transfer):
        return exp(-pi*(q-1)**2) * exp(-pi*(transfer-2)**2)
    right = integrate(lambda q: polarized(q, 3-q), 12, 384)
    wrong = integrate(lambda q: polarized(q, 3+q), 12, 384)
    assert right > 100*wrong > 0
    scale = 10**15
    print('Rigorous coefficient interval (units 1e-15):',
          (lo*scale).__floor__(), (hi*scale).__ceil__())


if __name__ == '__main__':
    main()
