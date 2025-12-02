# cgm

Project Title:  Enhanced Raman Light - Needle(ss) Glucose Monitoring 
Diabetes affects hundreds of millions worldwide, yet continuous glucose monitoring (CGM) still depends on invasive sensors that puncture the skin. Since the 1990s, researchers at MIT, Harvard, and elsewhere have explored Raman spectroscopy as a noninvasive alternative, using light scattering to capture glucose’s unique vibrational signature. The challenge is that glucose peaks are faint and often buried in complex tissue spectra dominated by water, proteins, and lipids. Recent advances in physics, such as depth-selective Raman using microscopic spatially offset Raman spectroscopy (mµSORS, Nature Biomedical Engineering 2021), show that smarter optics can enhance glucose signals by probing beneath the skin surface. 
My project builds on the latest advances in depth-selective Raman spectroscopy (mµSORS) and borrows ideas from coherent anti-Stokes Raman scattering (CARS). While CARS is a powerful but complex laser-based technique that boosts weak Raman signals and suppresses background noise, I aim to simulate some of those benefits in software rather than hardware. I will generate layered “tissue-like” spectra where surface signals dominate and deeper glucose peaks are faint, and then test whether simple AI models such as multivariate regression can separate surface from deep contributions and estimate glucose concentration. I’m modeling glucose signals in tissue while simulating different skin tones (light → dark) with varying fluorescence backgrounds.I’ll use meta-learning so the model adapts in a few samples per person, overcoming skin-color–driven spectral variation. Inspired by CARS, I will explore algorithmic “signal boosting” for glucose peaks and “background suppression” to reduce interference from water, lipids, and proteins. To probe robustness, I will add systematic stress tests: noise sweeps, baseline drift, wavenumber jitter, and background shifts in collagen:lipid:water ratios. In addition, I will run feature selection to identify a minimal set of Raman peaks (as few as 3–5 wavenumbers) that still predict glucose accurately, and compare performance against full-spectrum and random-peak baselines. Finally, I will use multi-offset spectra (two or more collection points) to mathematically “unmix” surface and deep signals, showing how accuracy improves for the deeper glucose contribution. By mapping when the models work, what breaks them, and how little spectral information is needed, this project highlights both the promise and limitations of AI-enabled Raman CGM—and shows how CARS-inspired ideas could make mµSORS simpler, cheaper, and more practical for noninvasive diabetes monitoring.
Question/Purpose: 
Can Physics Informed AI models inspired by coherent anti-Stokes Raman scattering (CARS) improve the accuracy of noninvasive glucose prediction from simulated depth-selective Raman (mµSORS) spectra by enhancing glucose signal and suppressing background noise
Hypothesis: 
If Raman spectra are processed using AI methods that simulate CARS-like signal boosting and background suppression within a physics-informed neural network (PINN) framework, where the CARS forward model—including nonlinear polarization, phase matching, and non-resonant background terms—is embedded as a differentiable constraint, and if meta-learning is applied to adapt across varying skin tones, fluorescence levels, and tissue conditions, then glucose peaks from deeper tissue layers can be extracted more accurately. This will lead to improved and personalized glucose prediction performance compared to purely data-driven, non-physics baselines.
Null hypothesis (H0)
A PINN with an embedded differentiable CARS forward model (including nonlinear polarization, phase matching, and NRB) plus meta-learning does not improve deep-layer glucose peak extraction or glucose prediction performance compared with physics-agnostic, purely data-driven baselines.
Independent Variable (IV):
   Light parameters:
      Raman peak intensities. Intensity values at selected wavenumbers 
   Tissue parameters:
      Skin tone (light, dark and others)
      Noise level
      Collagen:lipid:water ratios,
      Signal boosting / background suppression parameters
   AI parameters:   
      AI model type and training method (e.g., multivariate regression, meta-learning)
Control Variables:
   Glucose peak positions
   Consistent simulation conditions for each test
   Spectral resolution & laser wavelength
   Baseline noise and fluorescence kept constant for non-glucose signals
Dependent Variable (DV): 
The predicted glucose concentration in tissue (mg/dL or relative units). The accuracy of this prediction is evaluated by comparing it against the known ground truth glucose values used in the simulation.
Materials:
   Computer with Python or MATLAB
   AI/ML libraries
   Code to simulate layered tissue spectra
   Tools for feature selection like PCA and multi-offset spectral analysis in data
Research Summary:
Continuous glucose monitoring (CGM) remains invasive, requiring sensors that puncture the skin. Over the past three decades, researchers have explored Raman spectroscopy as a noninvasive alternative because it captures a molecular fingerprint of glucose based on light scattering (Scholtes-Timmerman et al., 2015). However, the main limitation has been that glucose peaks are faint and buried under stronger background signals from water, lipids, and proteins (Min et al., 2025). Recent advances in depth-selective Raman techniques, such as microscopic spatially offset Raman spectroscopy (mµSORS), have improved sensitivity by probing below the skin surface, demonstrating measurable glucose signals through tissue (Zhang et al., 2025). Still, mµSORS remains limited by optical complexity and spectral overlap, especially across different skin tones and fluorescence backgrounds. Meanwhile, researchers in biomedical AI have shown that machine learning and regression models can extract hidden Raman features to classify or estimate glucose concentrations, but most studies rely on single, shallow models that don’t generalize across conditions (Quang et al., 2024). Inspired by coherent anti-Stokes Raman scattering (CARS) — a nonlinear laser method that boosts weak signals and suppresses noise — this project proposes a software-based “virtual CARS” approach, applying AI to amplify glucose peaks and subtract background interference. By simulating layered, tissue-like spectra under varying skin tones, noise levels, and optical offsets, the project tests whether AI can reproduce the benefits of complex hardware through computation. This directly addresses the gap between optical hardware innovation and practical, affordable noninvasive CGM — justifying the question of whether CARS-inspired algorithms can enhance mµSORS-style Raman data for glucose prediction.
Safety/Ethics:
Entirely software-based; no humans or animals involved, fully safe and ethical.
Expected Outcome:
AI-based “virtual CARS” processing will improve non-invasive continuous glucose monitoring (cgm) detection accuracy. The project will also identify a minimal Raman fingerprint (3–5 key peaks) sufficient for reliable glucose prediction.

CARS-Inspired Enhancements (Innovation Summary)
In my project, I draw inspiration from several key principles of Coherent Anti-Stokes Raman Scattering (CARS) and translate them into software-based analogies. Just as CARS achieves coherent signal boosting by amplifying weak molecular vibrations, my model mimics this by giving extra weight to known glucose peaks during regression, enhancing their prominence over noise. CARS also uses multi-frequency excitation, combining multiple laser wavelengths to probe molecules more effectively; in my simulation, I replicate this by merging spectra from multiple spatial offsets or wavelengths to better isolate the glucose signal from deeper tissue layers. To address unwanted fluorescence and tissue background, I apply algorithmic background suppression through subtraction or orthogonal regression—an AI equivalent of CARS’ ability to cancel out noise optically. Similarly, where CARS achieves targeted molecular selectivity by tuning to specific chemical bonds, my model focuses on a minimal set of glucose fingerprint peaks identified through feature selection. Finally, I incorporate nonlinear enhancement using meta-learning so the AI can adapt to individual skin tones and tissue differences, simulating how CARS dynamically corrects for nonlinear variations in optical response. Together, these strategies reimagine complex CARS physics as a set of intelligent computational tools for enhancing Raman-based glucose detection.
References:
{very latest} https://gwern.net/doc/biology/2025-zhang.pdf?utm_source=chatgpt.com
https://arxiv.org/pdf/2510.06020 
{Latest} Yifei Zhang et al Subcutaneous depth-selective spectral imaging with mμSORS enables noninvasive glucose monitoring https://www.nature.com/articles/s42255-025-01217-w 
Shunhua Min, Haoyang Geng, Yuheng He, Tailin Xu, Qingzhou Liu, and Xueji Zhang 
Minimally and non-invasive glucose monitoring: the road toward commercialization
https://pubs.rsc.org/en/content/articlehtml/2025/sd/d4sd00360h
Tri Ngo Quang, Thanh Tung Nguyen & Huong Pham Thi Viet  Machine Learning Approach for Early Detection of Diabetes Using Raman Spectroscopy https://link.springer.com/article/10.1007/s11036-024-02340-w
Maarten J Scholtes-Timmerman, Sabina Bijlsma, Marion J Fokkert, Robbert Slingerland, Sjaak J F van Veen Raman Spectroscopy as a Promising Tool for Noninvasive Point-of-Care Glucose Monitoring https://pmc.ncbi.nlm.nih.gov/articles/PMC4455378/
Kaggle dataset - https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes/code
https://www.dexcom.com/all-access/dexcom-cgm-explained/direct-to-apple-watch
