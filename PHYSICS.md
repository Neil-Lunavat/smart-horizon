# The physics, derived

Everything the model assumes, worked out from first principles. These are consequences of how orbits and coordinate frames work, and they would be true of any GNSS dataset ever recorded.

---

## 1. The only free information in the problem

The competition supplies five columns, a timestamp and four error values, and
one extra fact: **whether the satellite is GEO or MEO.**

That one word is worth more than the five columns, because it pins down the
satellite's orbital period, and the orbital period determines every rhythm the
error can contain. Everything below follows from it.

---

## 2. A geosynchronous satellite repeats every sidereal day

A satellite is _geosynchronous_ when its orbital period equals one rotation of
the Earth relative to the fixed stars. That rotation is the **sidereal day**, not
the 24-hour solar day:

$$
T_{\text{sid}} = 86164.0905\ \text{s} = 23^{\text{h}}\,56^{\text{m}}\,04^{\text{s}}
$$

The 3 minutes 56 seconds of difference exists because the Earth also travels
around the Sun. In one year it makes one extra turn relative to the Sun than
relative to the stars, so

$$
\frac{1}{T_{\text{sid}}} = \frac{1}{T_{\text{solar}}} + \frac{1}{T_{\text{year}}}
\quad\Longrightarrow\quad
T_{\text{sid}} = \frac{86400 \times 365.25}{366.25} \approx 86164\ \text{s}
$$

From Kepler's third law, the radius that produces this period is

$$
a = \left(\frac{\mu\,T_{\text{sid}}^{2}}{4\pi^{2}}\right)^{1/3}, \qquad
\mu = 3.986004418\times10^{14}\ \text{m}^{3}\text{s}^{-2}
$$

$$
a \approx 42{,}164\ \text{km}
$$

> **What this means.** A geosynchronous satellite traces out the same path over
> the ground every 23h 56m, and so does any error in its predicted position. A
> seven-day window therefore contains **7.02 complete repeats**. We do not have
> to _search_ for the rhythm; we are told it exactly. That is why GEO is the
> easier of the two classes.
>
> **Why 23h 56m and not 24h matters:** over seven days the two drift apart by
> 28 minutes. A model that assumed a 24-hour cycle would be nearly half an hour
> out of phase by the end of the window, enough to turn a good fit into a bad one.

---

## 3. MEO: the period is known per constellation, but we are not told which

Medium-Earth-orbit satellites are designed so their ground tracks repeat over a
whole number of days:

| constellation | design                              | period $T$ | in hours |
| ------------- | ----------------------------------- | ---------- | -------- |
| GPS           | 2 revolutions per sidereal day      | 43 082 s   | 11.97 h  |
| Galileo       | 17 revolutions per 10 sidereal days | 50 567 s   | 14.05 h  |
| BeiDou MEO    | 13 revolutions per 7 sidereal days  | 46 393 s   | 12.89 h  |

> **What this means.** The competition tells us "MEO" but not _which_
> constellation, and the three periods differ by up to 17%. So unlike GEO, the
> period has to be **estimated from the window itself**, but only within the
> narrow band 38 000–54 000 s that physics allows. We are not searching blindly;
> we are searching a corridor two hours wide.

---

## 4. The central result: in Earth-fixed coordinates, x and y split in two

This is the derivation the whole MEO model rests on.

**Setup.** Orbit errors are naturally described in the _orbital_ frame, radial,
along-track, cross-track. An error that is roughly constant in that frame appears
to rotate with the satellite, at the orbital angular rate

$$
\omega_{o} = \frac{2\pi}{T}
$$

But the data is given in **ECEF**, Earth-Centred, Earth-Fixed, a frame that is
bolted to the rotating planet, turning at

$$
\omega_{e} = \frac{2\pi}{T_{\text{sid}}}
$$

**Where the axes actually point.** This matters for reading everything below,
and it is not a convention we chose. ECEF is the standard frame for GNSS products:

| axis | points at | rotates with the Earth? |
| ---- | --------- | ----------------------- |
| $z$  | the North Pole, along the Earth's spin axis | no, it **is** the spin axis |
| $x$  | the equator at 0 degrees longitude, the Greenwich meridian | yes |
| $y$  | the equator at 90 degrees East longitude | yes |

So $x$ and $y$ span the **equatorial plane** and $z$ is perpendicular to it.
The origin is the centre of mass of the Earth. Nothing here points "up" from the
satellite or along its direction of travel, which is the usual first guess. The
frame is anchored to the planet, not to the spacecraft.

The asymmetry that drives this whole section follows immediately. The Earth spins
about $z$, so $x$ and $y$ are dragged around once per sidereal day while $z$ is
left untouched. **That single fact is why z behaves differently from x and y in
every plot in this repository**, and it is a property of the coordinate system
rather than anything the satellite is doing.

One consequence is worth stating plainly: an error of "1 metre in $x$" is not a
fixed physical direction. Twelve hours later that same $x$ axis points the opposite
way in space. This is exactly why an orbital wobble that is a single clean tone in
space arrives in the file as two tones.

**The transformation.** Converting from an inertial frame to ECEF is a rotation
about the $z$ axis by angle $\theta(t) = \omega_e t$:

$$
\begin{pmatrix} x_{\text{ecef}} \\ y_{\text{ecef}} \\ z_{\text{ecef}} \end{pmatrix}
=
\begin{pmatrix}
\cos\theta & \sin\theta & 0 \\
-\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix} x_{\text{inertial}} \\ y_{\text{inertial}} \\ z_{\text{inertial}} \end{pmatrix}
$$

**Look at the last row.** It is $z_{\text{ecef}} = z_{\text{inertial}}$, the
rotation is _about_ the $z$ axis, so $z$ is left completely alone.

**Now the first two rows.** An error oscillating at the orbital rate contributes a
term like $A\sin(\omega_o t + \phi)$ to the inertial $x$ and $y$. After the
rotation, the ECEF $x$ contains products such as

$$
A\sin(\omega_{o}t + \phi)\cos(\omega_{e}t)
$$

Apply the product-to-sum identity:

$$
\sin\alpha\cos\beta = \tfrac{1}{2}\big[\sin(\alpha-\beta) + \sin(\alpha+\beta)\big]
$$

$$
\boxed{\;A\sin(\omega_{o}t+\phi)\cos(\omega_{e}t)
= \tfrac{A}{2}\sin\big((\omega_{o}-\omega_{e})t+\phi\big)
+ \tfrac{A}{2}\sin\big((\omega_{o}+\omega_{e})t+\phi\big)\;}
$$

One tone in, **two tones out**, at the difference and the sum of the two
frequencies. Converting back to periods:

$$
\boxed{\;P_{\pm} = \frac{1}{\left|\dfrac{1}{T} \pm \dfrac{1}{T_{\text{sid}}}\right|}\;}
$$

**Evaluated for each constellation:**

| constellation | $T$     | $P_{-}$ (sum of rates) | $P_{+}$ (difference) |
| ------------- | ------- | ---------------------- | -------------------- |
| GPS           | 11.97 h | **7.98 h**             | **23.93 h**          |
| Galileo       | 14.05 h | **8.85 h**             | **34.00 h**          |
| BeiDou MEO    | 12.89 h | **8.38 h**             | **27.92 h**          |

> Because the ground is spinning underneath the satellite, the x and y error channels do not oscillate at the orbital period at all, they oscillate at **two different periods, neither of which is the orbital period**, while z keeps the orbital period untouched.

> A model that fits the same rhythm to all three axes is guaranteed to be wrong on  two of them. This is exactly what we observe: a fixed-period model does fine on z  and badly on x and y. Once the two beat periods are used instead, x and y are recovered.

> $P_{+} = 23.93$ h, the sidereal day. That is the textbook fact that GPS ground tracks repeat once per sidereal day, falling straight out of the algebra. It is a check that the derivation is right.

> **The GEO case is the same equation.** Set $T = T_{\text{sid}}$ and the difference frequency goes to zero while the sum gives $T_{\text{sid}}/2$, the beats collapse back onto the sidereal day and its harmonics. GEO and MEO are not two different models; they are one model at two points of the same formula.

---

## 5. The clock is not periodic, and cannot be made to be

A satellite clock's error has three sources, and none of them is a rhythm:

**Random walk.** Frequency noise integrates into phase error. Over an interval
$\tau$ the accumulated error grows as

$$
\sigma(\tau) \propto \sqrt{\tau}
$$

It's a Wiener process. Its autocorrelation decays with no periodic component.

**Deterministic ageing.** A slow, near-linear frequency drift. This _is_
predictable, but it is already removed by the broadcast clock polynomial
$a_0 + a_1 t + a_2 t^2$ before the error we are given is computed. What remains
is the part the polynomial failed to capture.

What it's broadcasting is also it's own prediction about where it's going to be. That has inbuilt polynomial prediction, remodelling it makes no sense because we're looking at the residual.

**Step resets.** When the ground segment uploads fresh coefficients, the error
jumps. The uploads happen on an operational schedule that is not in the data.

> **What this means.** There is no daily cycle in the clock channel to find, so
> the model is **forbidden** from fitting harmonics to it and is restricted to
> level estimators. This is a restriction we impose deliberately: giving the
> model more freedom here makes it worse, because it fits noise and then
> extrapolates it into day 8.
>
> **The delivered data agrees.** `DATA_MEO_Train2.csv` has 143 rows but only
> **70 distinct clock values**, one repeating 17 times. It is a piecewise-constant
> step process, visibly a sequence of resets, not a drifting curve.

---

## 6. Why the offset of the window cannot be recovered

Suppose the seven days sit on an unknown constant $\varepsilon$. Nothing inside
the window can determine it, because there is no absolute reference to compare
against, every measurement is relative to the same unknown.

For a periodic component this does not matter: the oscillation is measured about
whatever level it sits on, and the level cancels. But for the **clock**, where
there is no oscillation, the level _is_ the entire signal. And the level of day 8
is set by resets that have not happened yet at prediction time.

> **What this means.** For the clock channel, the best any method can do is
> estimate the current level and carry it forward. That is precisely what the
> model does, and it is why no clever technique beats a well-chosen average
> there, a limit of the problem, not of the method.

---

## 7. What the model does with all of this

| physical fact                              | consequence in the model                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| GEO repeats at $T_{\text{sid}} = 86164$ s  | GEO basis fixed to the sidereal day and its harmonics; no period search          |
| a 7-day window holds 7.02 GEO cycles       | enough cycles to fit amplitude and phase reliably                                |
| MEO $T$ is 38 000–54 000 s and unknown     | $T$ estimated per window by Lomb-Scargle, inside that band only                  |
| ECEF splits x, y into $P_\pm$; z keeps $T$ | separate bases per axis; beats derived by formula, never searched                |
| clock is a random walk with resets         | harmonics barred on the clock; level estimators only                             |
| the window's offset is unidentifiable      | no attempt to predict a level shift; the current level is carried forward        |
| a true GEO satellite barely moves in ECEF  | flat estimators kept in the GEO candidate set, since the error can be pure drift |

---

## 8. Constants used

| symbol           | value                            | source                                       |
| ---------------- | -------------------------------- | -------------------------------------------- |
| $T_{\text{sid}}$ | 86164.0905 s                     | definition of the sidereal day               |
| $\mu$            | $3.986004418\times10^{14}$ m³/s² | WGS-84 Earth gravitational parameter         |
| $T_{\text{GPS}}$ | 43 082 s                         | 2 revs/sidereal day, by constellation design |
| $T_{\text{GAL}}$ | 50 567 s                         | 17 revs/10 sidereal days                     |
| $T_{\text{BDS}}$ | 46 393 s                         | 13 revs/7 sidereal days                      |

None of these is fitted, measured, or tuned. They are published constants that
define the constellations, and they are the entire physical input to the model.
