"""Independent continuous integrals for the unreduced SU(2) magnetic packet.

Run with python3 scripts/check_qym_magnetic_packet.py from any directory.
The integration nodes evaluate packet integrals; they do not define a spatial lattice.
"""
from fractions import Fraction
import json
from math import cos, exp, fsum, isqrt, pi, sin, sqrt
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


def check_local_gauge_projection():
    """Integrate the omitted radial profile directly with the 3D volume element."""
    def profile(r):
        return exp(-pi * r*r)

    def radial(f, panels):
        return integrate(lambda r: 2*pi*r*r * f(profile(r)), 6, panels)

    for panels in (192, 384):
        norm_sq = radial(lambda w: w*w, panels)
        projection = radial(lambda w: w**3, panels) / norm_sq
        exact_two_steps = radial(lambda w: w**4, panels) / norm_sq
        residual = radial(lambda w: (w*w - projection*w)**2, panels) / norm_sq
        assert abs(projection - (2/3)**1.5) < 2e-14
        assert abs(residual - (sqrt(1/8) - 8/27)) < 2e-14
        assert abs(exact_two_steps - projection**2 - residual) < 2e-14
    print(f'Local gauge rotation: omitted squared norm per unit input = {residual:.15f}')


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


def check_packet_form_comparison():
    """Rational bounds on the continuous Gaussian coefficients, before Hermite truncation."""
    pi_lo, pi_hi = Fraction(314, 100), Fraction(22, 7)
    root_lo, root_hi = Fraction(140, 99), Fraction(99, 70)
    assert root_lo**2 < 2 < root_hi**2
    decay = Fraction(1, 500)
    assert exp_negative_bounds(2*pi_lo)[1] < decay
    cubic_lo, cubic_hi = exact_enclosure()
    assert 0 < cubic_lo <= cubic_hi < Fraction(49, 1000)
    # In w_XY four zero-frequency terms of the sixteen cosine channels survive,
    # each with mass 1/8. The remaining Fourier weights are positive.
    wall_lo = Fraction(1, 32)
    seed_hi = Fraction()
    for j, frequency in enumerate((6, 12, 19), 1):
        mass_lo = 1 / (4 * root_hi)
        mass_hi = (1 + decay) / (4 * root_lo)
        if j == 3:
            mass_lo /= 4*pi_hi
            mass_hi /= 4*pi_lo
        transverse = 3 if j == 3 else 1
        ratio_lo = (1+transverse)*pi_lo + 4*pi_lo**2*j*j / (1+decay)
        ratio_hi = (1+transverse)*pi_hi + 4*pi_hi**2*j*j
        if j == 3:
            ratio_lo -= cubic_hi**2 / (wall_lo*mass_lo)
        assert frequency**2 < ratio_lo
        seed_hi += 1/(8*mass_lo) + mass_hi*ratio_hi/2
    # For the upper bound discard cos² <= 1 in both wall integrals.
    # At g=2 the Gaussian seed has E[XYZ]=0 and E[X²Y²]=E[X²Z²]=1.
    seed_hi += 2*(Fraction(1, 8) + 1/(64*pi_lo))
    assert seed_hi < 32
    print('Three-profile form floor >= 37/2 for all g; Gaussian seed energy < 32 at g=2')


def check_local_gauss_trials():
    """Integrate raw-amplitude shifts and the two-profile transverse energy."""
    def bell(x, width):
        return exp(-pi*(x/width)**2)

    def longitudinal(x, width):
        return bell(x, width)*cos(2*pi*x/width)

    fixed_product = None
    for width in (1, 2, 4, 8):
        def overlap(parameter_width):
            # omega = (x_1 / parameter_width) f_parameter_width(x).
            radius = 6*min(width, parameter_width)
            def xpart(x):
                y = x/parameter_width
                return bell(x, width)*bell(x, parameter_width)*(1-2*pi*y*y)/parameter_width
            xx = integrate(xpart, radius, 768)
            yy = integrate(lambda y: bell(y, width)*bell(y, parameter_width), radius, 768)
            zz = integrate(lambda z: longitudinal(z, width)*longitudinal(z, parameter_width),
                           radius, 768)
            return xx*yy*zz

        mass = (integrate(lambda x: bell(x, width)**2, 6*width, 768)**2 *
                integrate(lambda z: longitudinal(z, width)**2, 6*width, 768))
        matched = width * overlap(width) / mass
        fixed = width * overlap(1) / mass
        assert abs(matched - 0.5) < 2e-13
        assert abs(fixed) <= 1 / ((mass/width**3) * width**4)
        print(f'L={width} standardized shift: matched={matched:.8f} fixed={fixed:.8f}')

        def radial(f):
            return integrate(lambda r: 2*pi*r*r * f(r/width), 6*width, 768)
        # The angular averages below include both Cartesian polarizations of the curl.
        metric = radial(lambda r: (2/3)*r*r*bell(r,1)**2)
        curl = radial(lambda r: (4 - 16*pi*r*r/3 + 8*pi*pi*r**4/3)*bell(r,1)**2)/width**2
        wall = radial(lambda r: r**4*bell(r,1)**4/3)
        proj = radial(lambda r: (2/3)*r*r*bell(r,1)**3) / metric
        leak = radial(lambda r: (2/3)*r*r*bell(r,1)**2*(bell(r,1)-proj)**2) / metric
        assert abs(leak - (2**(-2.5) - (2/3)**5)) < 2e-14
        assert abs(curl*width**2/metric - 5*pi) < 1e-12
        assert abs(wall/width**3 - 5/(512*pi*pi)) < 1e-14
        # Magnetic curvature excess when the spatially varying colour angle is
        # replaced by its compression. A=u(x/L)/L, gauge time 1 and coupling 2.
        grad_wall = radial(lambda r: 4*pi*pi*(2/3)*r**4*bell(r,1)**4) / width**4
        assert abs(2*grad_wall*width - 5/32) < 2e-14
        # Metric of the exact one-parameter orbit at unit width. Integrate the
        # square of the raw velocity, then check the minimizing orbit velocity.
        j_metric = radial(lambda r: 4*pi*pi*r*r*bell(r,1)**2) / width**3
        m_unit = metric / width**3
        second = leak + proj**2
        for colour_speed in (0, 1, 3):
            twist = 2 * colour_speed
            orbit_metric = j_metric + m_unit*second*twist**2
            sigma = -m_unit*proj*twist / orbit_metric
            horizontal = (radial(lambda r: (2/3)*r*r*bell(r,1)**2 *
                                 (1 + twist*sigma*bell(r,1))**2) / width**3
                          + j_metric*sigma**2)
            assert abs(horizontal - (m_unit - (m_unit*proj*twist)**2/orbit_metric)) < 2e-14
            assert m_unit*(1-proj**2/second) <= horizontal <= m_unit+2e-14
        # Six independent amplitudes of standard deviation 1/L, g=2.
        energy = 6*width**2/(8*metric) + 3*curl/width**2 + 12*wall/width**4
        if fixed_product is None:
            fixed_product = energy
        assert abs(width*energy - fixed_product) < 1e-11
        print(f'L={width} transverse energy={energy:.10f} rotation loss={leak:.10f}')


def main():
    reference = Path(__file__).resolve().parents[1] / 'docs/fixtures/qym_magnetic_packet.json'
    record = json.loads(reference.read_text())
    assert record['schema'] == 'qym-cubic-continuous-packet/v1' and record['coupling'] == 1
    check_local_gauss_trials()
    check_packet_form_comparison()
    check_local_gauge_projection()
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
