---
title: "Prior Work on the Flight-Height Equation"
subtitle: "A literature review of the flight-height report, and what it should cite"
author: "Zezhang Yang"
date: "September 20, 2026"
---

# Summary

This review asks whether the flight-height equation of the companion report, *A Flight-Height Equation for Drone Waterfowl Surveys*, has been published before. It searched the drone wildlife survey literature, the remote-sensing image-quality literature and the small-object detection literature, and retrieved the full text of the closest papers.

**The equation is not new as geometry, and the idea behind it is about sixty-five years old.** Two of its parts are already standard and should be cited rather than derived:

- Equation (2), the ground-sampling-distance relation, is textbook photogrammetry and is built into every drone flight-planning application.
- The *form* of Equation (1) — fix the number of pixels the target must span, then solve the geometry for height — was published for wildlife drone surveys by Burke et al. (2019), who also provide a public web calculator for it.

**Two parts were not found anywhere.** Both concern where the constants come from, not the algebra:

1. **An optimum pixel size measured from a detector's own response curve and then used to set a flight height.** Published work either assumes a threshold in advance, or measures accuracy against image resolution and stops there. No paper found inverts a measured curve into a survey parameter.
2. **The blur correction expressed as a single multiplier on flight height.** The ingredients are known — adding blur widths in quadrature is routine in astronomy and microscopy, and the General Image-Quality Equation has divided resolution by sharpness since 1997 — but reducing blur to one bounded number, *k* = 1.09, that needs no field measurement has no precedent that this review could find.

The contribution is therefore **calibration and closure, not derivation**. Stated so that it survives review: *others said "fly so the animal spans N pixels"; this report measures what N is for a convolutional detector, shows that N is a peak rather than a floor, and corrects it for blur.*

**One finding may change a number rather than a claim.** The closest prior study, Viegut et al. (2024), is from this laboratory and uses this corpus. It separates *availability* — whether a bird is visible in the image at all — from *perception*, whether the algorithm finds a visible bird, and reports that availability falls as ground sampling distance grows, while perception shows no such pattern. The optimum of 18 pixels in the companion report is a perception optimum, because it is fitted to birds that were annotated. The operational optimum is the product of the two, and a declining availability multiplied by a peaked perception curve places it **below 55 m** over vegetated sites. Section 7 sets out what to do about it.

**Main recommendations.** Add a related-work section; soften the novelty claim in the summary from giving an equation to calibrating one; address availability in the limitations or, better, in the result; and cite Brown et al. (2022), whose aperture experiment is stronger evidence for the blur correction than the report's own blur ladder.

# 1. Scope and method

## 1.1 What counts as prior art here

The companion report's equation is not a single claim but five, which can be published independently and, as it turns out, mostly have been:

1. the imaging geometry that converts height into ground sampling distance;
2. the inversion of that geometry, which fixes a required size in pixels and solves for height;
3. the choice of the required size — where the number comes from and what it means;
4. the treatment of blur, which widens a box by a fixed number of pixels regardless of height;
5. the operational closure — take-off offset, camera constants from EXIF data, species table, validation.

A finding that part of the equation is already published does not weaken the report, provided the report cites it. What would weaken the report is a claim of novelty over something standard, which a reviewer will notice immediately. This review therefore reports prior art claim by claim rather than reaching a single verdict.

## 1.2 What was searched

Nine search families were run, and the full text of the four closest papers was retrieved and read:

| Search family | Purpose |
|---|---|
| GSD and altitude equations for animal surveys | Find the equation itself |
| Minimum pixels on target for detection | Find the origin of the pixel criterion |
| Detection accuracy against spatial resolution and PSF | Find work measuring what Section 6 measures |
| Optimal UAV altitude: wildlife, agriculture, search and rescue | Find the inversion in other fields |
| Johnson criteria and DRI practice | Establish the classical precedent |
| GIQE and NIIRS | Establish the image-quality precedent |
| Quadrature combination of blur widths | Test the novelty of Equation (5) |
| Waterfowl and waterbird drone surveys | Find operational practice and competing recommendations |
| The Missouri group's own output | Find work on this corpus |

Coverage is good for the wildlife and remote-sensing literatures and for computer vision. It is weaker for defence and surveillance work, where target-acquisition modelling is extensive and partly unpublished, and for the grey literature of survey protocols issued by agencies. Neither gap is likely to contain a closed-form height equation calibrated to a neural detector, but neither can be excluded.

# 2. The equation, claim by claim

| Claim in the companion report | Status | Closest prior work |
|---|---|---|
| Eq. (2): GSD = *Hw* ÷ (*fN*) | Standard | Photogrammetry texts; flight-planning software |
| Eq. (3): focal length in pixels as one camera constant | Standard; equivalent to an angular pixel scale | Burke et al. (2019), Eqs. 2–3 |
| Eq. (1) form: fix pixels on target, solve for *H* | Published, for this exact purpose | Burke et al. (2019), §2 |
| Pixels on target as the criterion for detectability | Classical, from 1958 | Johnson (1958); DRI practice |
| *P*~optimum~ = 18 px, measured from a fitted response curve with bootstrap intervals | Not found | Brown et al. (2022); Shermeyer & Van Etten (2019); Jaffe et al. (2019) |
| The detection rate has an interior peak, not a floor | Not found for wildlife | Scale-mismatch literature explains the shape, not the optimum |
| Eq. (5): geometric size and blur margin add in quadrature | Standard technique, new application | PSF and deconvolution practice |
| Eqs. (8)–(9): blur as a single height multiplier from κ alone | Not found | GIQE's GSD ÷ RER ratio is the nearest relative |
| Eq. (11): motion blur as an increase in effective target size | Known as smear, handled differently | Aerial survey planning practice |
| Bounds: disturbance floor and the 122 m ceiling | Standard; the floor has a published method the report does not use | Duporge et al. (2021); 14 CFR §107.51 |

Table: Table 1. Prior-art status of each claim in the flight-height equation.

# 3. Prior work that already contains part of the equation

## 3.1 The geometry is textbook

The relation between flight height, focal length, sensor width, image width and ground sampling distance is standard photogrammetry, and is implemented in every drone mapping application. Ecological papers that use it state it without derivation and cite nothing. The companion report's Section 2.2 derives it from similar triangles, which is good exposition but should be introduced as a restatement, not a result.

## 3.2 Burke et al. (2019): the same inversion, a different constant

This is the closest prior art for Equation (1), and the one a reviewer is most likely to know.

Burke et al. define an angular pixel scale ρ~a~ = θ ÷ #pixels, where θ is the field of view, and a physical pixel scale on the ground ρ~p~ = *h* tan(ρ~a~) as a function of height *h*. They then invert this for height, given a required size of the target in pixels. Algebraically this is the companion report's Equations (2) to (4), with the field of view in place of the ratio of focal length to sensor width. Their worked example is structurally identical to the report's Section 3.3: an animal 1 m long, a requirement of 10 pixels, a FLIR Tau 640 camera with a 13 mm lens, giving a recommended height of **81.5 m** above ground level and a field of view of 67.5 × 54.5 m. They also publish an online calculator that performs it for an arbitrary camera.

**The difference lies in the constant, and it is a real difference.** Burke et al.'s requirement of 10 pixels is not a detection criterion. It is a radiometric one, derived from a mixed-pixel argument: for a circular source projected onto a square grid of pixels, the point at which pixels containing only the object outnumber pixels shared with the background occurs at a diameter of about 10 pixels, and at that size the mean recorded temperature is within 5% of the object's true temperature. The criterion answers the question *how large must an animal appear for its temperature to be measured correctly*, not *how large must it appear to be found*. The companion report's 18 pixels answers the second question, for a convolutional detector, from 38,370 annotated birds.

The two numbers are therefore not comparable, and the report must say so. A reader who knows Burke et al. will otherwise read 18 against 10 as an unexplained disagreement with a published result.

## 3.3 Johnson (1958) and the DRI criteria: the classical ancestor

The idea that detectability is set by how many pixels span a target dates to Johnson's work at the US Army Night Vision laboratory in 1957–58. Johnson's criteria state the resolution needed across a target's critical dimension for four tasks, each at 50% discrimination probability:

| Task | Line pairs across the target | Approximate pixels |
|---|--:|--:|
| Detection: an object is present | 1.0 ± 0.25 | 2 |
| Orientation | 1.4 ± 0.35 | 3 |
| Recognition: the class of object | 4.0 ± 0.8 | 8 |
| Identification: the specific object | 6.4 ± 1.5 | 13 |

Table: Table 2. Johnson's criteria, the classical precedent for a pixels-on-target requirement.

These remain the basis of the detection–recognition–identification range calculators used in surveillance and defence optics. They matter to the companion report in two ways.

First, they establish that the *form* of the argument is old, which is another reason to frame the contribution as calibration.

Second, they give a check on Section 3.5, which treats species identification as needing about 45 pixels against 18 for counting, a ratio of 2.5. Johnson's ratio between identification and detection is 6.4, much steeper. The difference is explicable — the report's detection criterion is a 94.6% detection rate for an algorithm, not a 50% threshold for a human observer, and telling one duck species from another is an easier discrimination than Johnson's task of identifying a specific vehicle — but the report should make the comparison rather than leave the reader to find it.

# 4. Prior work that measures what this report measures

Three strands of work measure detector performance against image resolution. None inverts the result into a recommended flight height, which is why none pre-empts the report; but all three strengthen it, and one of them tests the blur claim better than the report does.

## 4.1 Accuracy against resolution, with the optics modelled

**Brown et al. (2022)** degrade high-resolution aerial imagery of cattle across a range of ground sampling distances, modelling the optical point spread function explicitly for both circular and cassegrain apertures using Airy-disk diffraction, train a YOLOv5 detector at each degraded resolution, and measure the result. They find a sharp fall in performance at about 0.5 m per pixel for cattle-sized animals. Their most useful result for the companion report is a different one: **at matched ground sampling distance, changing the aperture type cost 52% of mAP.** Blur therefore sets the size requirement independently of pixel count, which is exactly the claim of the report's Section 6.4, demonstrated by a cleaner experiment than the report's ladder of added Gaussian blur.

Brown et al. also criticise earlier work for degrading ground sampling distance without degrading optical resolution to match, which produces unrealistically sharp images. The report's Table 11 is open to the mirror-image criticism: it adds blur without changing the ground sampling distance. The report already concedes that the two larger blur levels show direction rather than scale; citing Brown et al. here converts a weakness into a citation.

**Shermeyer & Van Etten (2019)** degrade 30 cm satellite imagery down to 4.8 m using a variable Gaussian blur to simulate the sensor's point spread function, and report mean average precision falling from 0.53 to 0.11, with the loss concentrated between 30 cm and 120 cm. The relevance is the method — Gaussian blur as a stand-in for the PSF, which is what the report's Section 6.4 does — and the finding that the decline is monotonic.

Neither paper reports an interior optimum. Both report degradation.

## 4.2 Designing for the algorithm rather than the observer

**Jaffe, Zelinski & Sakla (2019)**, in *IEEE Transactions on Geoscience and Remote Sensing*, make an argument one level above the companion report's. Remote sensing systems, they observe, have historically been designed against models of **human** image quality — the National Imagery Interpretability Rating Scale and the General Image-Quality Equation that predicts it. They argue the objective should instead be the accuracy of a convolutional network, and simulate ground sampling distance, modulation-transfer-function blur and noise to measure classification accuracy against pixels on target.

This is the companion report's thesis applied to sensor design rather than flight planning, and it gives the report a pedigree in the remote-sensing literature in one sentence.

It also makes the **GIQE** worth naming. The GIQE (Leachtenauer et al. 1997) predicts interpretability from three attributes: scale, expressed as ground sampling distance; sharpness, measured from the system's modulation transfer function as the relative edge response; and signal-to-noise ratio. Its spatial term is the **ratio GSD ÷ RER** — resolution divided by sharpness. That is structurally the same insight as the report's κ = *P*~optimum~ ÷ *b*: what matters is not resolution alone but resolution measured against blur. Saying so in Section 2.3 makes *k* = 1.09 look like an instance of an accepted principle rather than an arbitrary correction, which is how a sceptical reader will otherwise read it.

## 4.3 Quadrature addition of blur

Equation (5), which adds the geometric size and the blur margin in quadrature, is standard practice wherever a measured width must be separated from the instrument that blurred it — astronomy, microscopy and deconvolution generally, where the measured width of a source is routinely written as the square root of the sum of the squared intrinsic and instrumental widths. The report presents it as a modelling choice justified by the data, which is correct and honest, but it should note that the choice is conventional. It is easier to defend as convention than as invention.

# 5. Prior work on this corpus

**Viegut et al. (2024)**, in *Drones*, is the most important citation in this review. It is from this laboratory — Yi Shang is a co-author — and it uses this corpus: a DJI Mavic 2 Pro over Missouri Department of Conservation wetlands, at ground sampling distances of 0.38 to 2.29 cm per pixel, corresponding to 15 to 90 m of altitude.

Its framework is the one the companion report needs. It decomposes count error into two biases:

- **availability bias** — whether a bird is present in the image to be found at all, which depends on vegetation cover, sky condition and resolution;
- **perception bias** — whether the algorithm finds a bird that is available.

Its findings are:

- availability was best explained by the interaction of **vegetation cover type, sky condition and ground sampling distance**, with taller and more complex vegetation reducing availability as the ground sampling distance grew;
- the algorithm's correct detection of *available* birds showed **no pattern** in ground sampling distance, vegetation or sky condition;
- false positives were best explained by vegetation with features of similar size and shape to birds;
- corrected count error was 5.59%.

The companion report must cite this, both because it is the prior treatment of the same data and because a reader from the same group will know it. The difference should be stated plainly: Viegut et al. characterise bias as a function of ground sampling distance; the companion report inverts a detector's response into a planning height. But the overlap also creates a problem, which is the subject of the next section.

# 6. Current operational practice

Current practice is worth summarising, because it shows what the equation replaces and that the something it replaces is not principled.

- **Size classes chosen by eye.** Seabird surveys fly large species such as cormorants and great black-backed gulls at 40 m, medium species such as the common gull at 30 m, and small species such as terns and black-headed gulls at 20 m. This is the right shape of rule with no equation behind it.
- **Fixed targets for ground sampling distance.** Recent waterbird work reports best results at 25 m or lower, a ground sampling distance of 6.83 mm per pixel or finer; multi-species seabird surveys resample imagery to a standard 5 mm; dugong surveys settled on 2.5 cm per pixel.
- **Trial and error.** Several papers recommend, as method, varying the planned height in a mission-planning application until the resulting pixel size looks adequate.
- **The review-level verdict.** Axford et al. (2024), reviewing deep learning for animal detection in drone imagery, name the **required ground sampling distance** as an open question across studies. That sentence is the best available justification for the companion report existing, and belongs in its Section 1.

Note the distance between these numbers and the report's. A ground sampling distance of 6.83 mm per pixel puts a 40 cm waterbird at about 58 pixels, against the report's 18. Most of that gap is task — those studies identify species, the report counts birds — and Section 3.5 already handles the distinction. But the report recommends flying roughly three times higher than the nearest published waterbird recommendation, and should make that comparison explicitly rather than leaving a reader to notice it unaided.

# 7. The open issue: availability against perception

This is the one item in this review that may change a number in the companion report.

The optimum pixel size *P*~optimum~ is fitted in Section 6 to birds that appear in the annotations. A bird too small or too obscured to be annotated is not in the denominator. What Section 6 measures is therefore the probability that the detector finds a bird **given that the bird was available to be found** — perception, in Viegut et al.'s terms.

Viegut et al. report that availability itself declines as the ground sampling distance grows, most steeply over tall and complex vegetation. The probability that a bird present on the ground is counted is the product of the two:

*P*(counted) = *P*(available | height, cover, sky) × *P*(detected | available, pixel size)

The second factor peaks at 18 pixels and is flat around it. The first declines monotonically with height. Their product peaks **lower than the second factor alone**, by an amount that depends on the slope of the availability curve — which Viegut et al. estimate, for these sites and this drone. Over open water the correction should be small, because availability there is near one at all the heights flown; over moist-soil and flooded vegetation it will not be.

The companion report's Section 9 lists "detection rate only" as a limitation, but with reference to precision, not availability. Availability is the larger omission of the two.

A second-order version of the same problem affects the constants themselves. At greater heights a bird that is hard to annotate is also likely to be annotated less precisely, so both *P*~optimum~ and the blur margin *b* = 7.29 pixels partly record the annotator's behaviour as a function of height, not only the optics and the detector.

**What to do, in order of increasing effort and value:**

1. Name the limitation. One sentence in Section 9: *H*~rec~ optimises perception given availability; availability declines with height in tall vegetation (Viegut et al. 2024), so the operational optimum over vegetated sites is lower than 55 m.
2. Bound it. State the recommended height for open water, where the correction is small, and mark the vegetated case as unresolved.
3. Resolve it. Multiply Viegut et al.'s availability curve into the fitted response *R*(*P*~box~) and report a second, habitat-conditional height. This uses a published curve from the same sites and the same aircraft, so it needs no new fieldwork, and it turns the report's largest exposure into its most defensible result.

# 8. Anticipated questions

| Question | Where it comes from | Answer |
|---|---|---|
| Is this not just the GSD formula? | Any reader who plans drone flights | Concede the geometry in Section 2.1; answer with *k* and the measured *P*~optimum~, not by defending the algebra |
| Why 18 pixels when Burke says 10 and Johnson about 2? | A reader who knows either literature | Different criteria: radiometric accuracy, 50% human discrimination, 94.6% algorithmic detection. One sentence each |
| Is the peak real, or is it your training set? | Any computer-vision reader | Partly the latter, and the report should say so: birds of 15 to 25 pixels dominate training, the dip at 30–40 px and the recovery at 100 px look like scale-distribution effects. The concession costs nothing and the literature supports it |
| You recommend flying higher than anyone else | A waterbird biologist | Section 6's task distinction, plus the availability work of Section 7 — which must be done first |
| Where does κ come from? | A reviewer | One corpus, measured once. Already conceded in Section 9; pair it with the GIQE analogy so the form of the correction is not also in question |

Table: Table 3. Questions the companion report should answer before it is circulated.

# 9. Recommended changes to the companion report

In descending order of value:

1. **Add a related-work section** of about one page after Section 1, covering: the geometry as standard; Burke et al. as the prior inversion; Johnson and DRI as the classical criterion; Brown et al. and Shermeyer & Van Etten for resolution against accuracy; Jaffe et al. and the GIQE for designing to an algorithm; Viegut et al. for this corpus; Axford et al. for the research gap.
2. **Soften the novelty framing in the summary.** "This report gives drone operators an equation" reads as a claim that the equation is new. Calibrating and closing an existing equation is both more accurate and harder to attack.
3. **Address availability**, at minimum as a named limitation (Section 7, action 1), preferably as a result (action 3). This is the only recommendation here that could change a recommended height.
4. **Cite Brown et al. in Section 6.4.** Their aperture result is independent and cleaner evidence for the claim that blur, not pixel count alone, sets the size requirement.
5. **Place κ beside the GIQE's GSD ÷ RER ratio** in Section 2.3, so that the blur correction reads as an instance of an accepted principle.
6. **Use Duporge et al. (2021) for the disturbance floor** in Section 2.4, in place of "which biologists should set". Their audiogram method computes a lower bound from the species' hearing, which makes the bracket around *H*~rec~ symmetric: a derived floor and the §107.51 ceiling.
7. **Add the practice comparison** of Section 6 to the companion report's Section 1, including the explicit comparison with the 25 m waterbird recommendation.
8. **Note the conventional status of Equation (5)** in Section 2.2, in one clause.

# References

Verification note: each reference below was checked against the publisher's record or the paper's own front matter, with three exceptions. Johnson (1958) is a classical conference reference traced only through secondary sources, and its page range should be confirmed before it is cited in the companion report. The volume and page numbers of Brown et al. in *Computers and Electronics in Agriculture* were not confirmed; the preprint was the version read. Duporge et al.'s author list is given here in the publisher's abbreviated form and should be expanded from the paper.

Axford, D., Sohel, F., Vanderklift, M., & Hodgson, A. (2024). Collectively advancing deep learning for animal detection in drone imagery: Successes, challenges, and research gaps. *Ecological Informatics*, 83, 102842.

Brown, J., Qiao, Y., Clark, C., Lomax, S., Rafique, K., & Sukkarieh, S. (2022). Automated aerial animal detection when spatial resolution conditions are varied. *Computers and Electronics in Agriculture*. Preprint: arXiv:2110.01329.

Burke, C., Rashman, M., Wich, S., Symons, A., Theron, C., & Longmore, S. (2019). Optimising observing strategies for monitoring animals using drone-mounted thermal infrared cameras. *International Journal of Remote Sensing*, 40(2), 439–467. doi:10.1080/01431161.2018.1558372

Corcoran, E., Winsen, M., Sudholz, A., & Hamilton, G. (2021). Automated detection of wildlife using drones: Synthesis, opportunities and constraints. *Methods in Ecology and Evolution*, 12(6), 1103–1114.

Duporge, I., Spiegel, M. P., Thomson, E. R., et al. (2021). Determination of optimal flight altitude to minimise acoustic drone disturbance to wildlife using species audiograms. *Methods in Ecology and Evolution*, 12(11), 2196–2207. doi:10.1111/2041-210X.13691

Federal Aviation Administration. *Small Unmanned Aircraft Systems*, 14 CFR Part 107, §107.51 Operating limitations for small unmanned aircraft.

Jaffe, L., Zelinski, M., & Sakla, W. (2019). Remote sensor design for visual recognition with convolutional neural networks. *IEEE Transactions on Geoscience and Remote Sensing*. Preprint: arXiv:1906.09677.

Johnson, J. (1958). Analysis of image forming systems. *Proceedings of the Image Intensifier Symposium*. (Johnson's criteria; see also modern detection–recognition–identification practice.)

Leachtenauer, J. C., Malila, W., Irvine, J., Colburn, L., & Salvaggio, N. (1997). General image-quality equation: GIQE. *Applied Optics*, 36(32), 8322–8328. doi:10.1364/AO.36.008322

Shermeyer, J., & Van Etten, A. (2019). The effects of super-resolution on object detection performance in satellite imagery. *CVPR Workshops (EarthVision)*. Preprint: arXiv:1812.04098.

Viegut, R., Webb, E. B., Raedeke, A., Tang, Z., Zhang, Y., Zhai, Z., Liu, Z., Wang, S., Zheng, J., & Shang, Y. (2024). Detection probability and bias in machine-learning-based unoccupied aerial system non-breeding waterfowl surveys. *Drones*, 8(2), 54. doi:10.3390/drones8020054

Zhang, Y., Feng, Y., Wang, S., Tang, Z., Zhai, Z., Viegut, R., Webb, L., Raedeke, A., & Shang, Y. (2024). Deep learning models for waterfowl detection and classification in aerial images. *Information*, 15, 157.
