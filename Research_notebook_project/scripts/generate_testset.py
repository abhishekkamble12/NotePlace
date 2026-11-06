# scripts/generate_testset.py
"""
Generates the benchmark ~150 (claim, quote, context, label, perturbation_type, source)
dataset across 30 real technical, scientific, and medical quotes.
"""
import csv
from pathlib import Path

DATA = [
    # 1. Google Sycamore
    {
        "quote": "In October 2019, Google announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years.",
        "context": "Quantum computational supremacy marks a milestone where a programmable quantum device solves a problem no classical supercomputer can compute in feasible time. In October 2019, Google announced that their 53-qubit Sycamore processor had achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take a classical supercomputer 10,000 years. IBM later countered that with optimized tensor contraction, classical supercomputers could solve it in 2.5 days.",
        "source": "Nature 574, 505-510 (2019)",
        "variations": [
            ("Google announced quantum supremacy in October 2019 using their 53-qubit Sycamore processor.", "supports", "original"),
            ("Google's Sycamore processor featured 72 functional superconducting qubits in the 2019 benchmark.", "unsupported", "number"),
            ("In October 2023, Google first demonstrated quantum supremacy with the Sycamore chip.", "unsupported", "date"),
            ("Google's Sycamore processor permanently obsoleted all classical supercomputers across all computational domains.", "unsupported", "overgeneralize"),
            ("IBM announced quantum supremacy with the Sycamore processor in 2019.", "unsupported", "entity"),
        ]
    },
    # 2. Transformer Architecture
    {
        "quote": "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "context": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling. The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "source": "Vaswani et al., NeurIPS 2017",
        "variations": [
            ("The Transformer architecture computes input and output representations relying entirely on self-attention without recurrence or convolutions.", "supports", "original"),
            ("The Transformer relies primarily on a hybrid combination of depthwise convolutions and bidirectional recurrent units.", "unsupported", "entity"),
            ("The Transformer was introduced in 2012 as an extension to convolutional vision networks.", "unsupported", "date"),
            ("Every neural network published after 2017 abandoned recurrent layers completely.", "unsupported", "overgeneralize"),
            ("The Transformer uses self-attention mechanisms to process sequence data.", "supports", "original"),
        ]
    },
    # 3. Chinchilla Scaling Laws
    {
        "quote": "We find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled.",
        "context": "Recent work has investigated the scaling behavior of autoregressive language models. We find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. This predicts that large models such as Gopher are significantly undertrained.",
        "source": "Hoffmann et al., DeepMind 2022",
        "variations": [
            ("Compute-optimal training requires scaling model size and training tokens in equal proportion.", "supports", "original"),
            ("For compute-optimal scaling, the number of tokens must be quadrupled for every doubling of model parameters.", "unsupported", "number"),
            ("DeepMind discovered in 2015 that model parameters should scale four times faster than token count.", "unsupported", "date"),
            ("No model trained before 2022 was able to produce coherent English text.", "unsupported", "overgeneralize"),
            ("OpenAI researchers established the Chinchilla scaling law by testing smaller models on larger token sets.", "unsupported", "entity"),
        ]
    },
    # 4. AlphaFold 2
    {
        "quote": "In the CASP14 assessment, AlphaFold 2 produced structure predictions with a median backbone accuracy of 0.96 Å r.m.s.d., compared with 2.8 Å for the next best method.",
        "context": "Determining protein 3D structures from amino acid sequence alone has been a 50-year grand challenge in biology. In the CASP14 assessment, AlphaFold 2 produced structure predictions with a median backbone accuracy of 0.96 Å r.m.s.d., compared with 2.8 Å for the next best method, demonstrating near-experimental accuracy.",
        "source": "Jumper et al., Nature 2021",
        "variations": [
            ("AlphaFold 2 achieved a median backbone accuracy of 0.96 Å r.m.s.d. during the CASP14 assessment.", "supports", "original"),
            ("AlphaFold 2 achieved an accuracy of 0.12 Å r.m.s.d. in the CASP14 assessment.", "unsupported", "number"),
            ("In the 2010 CASP assessment, AlphaFold 2 solved the protein folding problem.", "unsupported", "date"),
            ("AlphaFold 2 eliminated the need for all future wet-lab experimental crystallography in structural biology.", "unsupported", "overgeneralize"),
            ("RosettaFold achieved a median backbone accuracy of 0.96 Å in CASP14.", "unsupported", "entity"),
        ]
    },
    # 5. CRISPR Cas9 Sickle Cell
    {
        "quote": "In December 2023, the US FDA approved Casgevy, the first cell-based gene therapy utilizing CRISPR/Cas9 technology for the treatment of sickle cell disease in patients 12 years and older.",
        "context": "Gene-editing advancements culminated in regulatory history when in December 2023, the US FDA approved Casgevy, the first cell-based gene therapy utilizing CRISPR/Cas9 technology for the treatment of sickle cell disease in patients 12 years and older. The treatment involves editing the patient's own hematopoietic stem cells.",
        "source": "FDA Regulatory Announcement (Dec 2023)",
        "variations": [
            ("Casgevy was approved by the US FDA in December 2023 as the first CRISPR/Cas9 gene therapy for sickle cell disease.", "supports", "original"),
            ("Casgevy was approved for patients as young as 2 years old in the initial December 2023 FDA authorization.", "unsupported", "number"),
            ("The FDA approved Casgevy for sickle cell disease back in December 2016.", "unsupported", "date"),
            ("CRISPR technology has completely eradicated sickle cell disease worldwide.", "unsupported", "overgeneralize"),
            ("The European Medicines Agency was the first to approve Casgevy in December 2023, preceding the FDA.", "unsupported", "entity"),
        ]
    },
    # 6. James Webb Space Telescope
    {
        "quote": "On July 11, 2022, NASA revealed Webb's First Deep Field, capturing galaxy cluster SMACS 0723 as it appeared 4.6 billion years ago, representing the highest-resolution infrared view of the universe to date.",
        "context": "Deploying 1.5 million kilometers from Earth at Lagrange point L2, the James Webb Space Telescope began scientific operations. On July 11, 2022, NASA revealed Webb's First Deep Field, capturing galaxy cluster SMACS 0723 as it appeared 4.6 billion years ago, representing the highest-resolution infrared view of the universe to date.",
        "source": "NASA Press Release, July 2022",
        "variations": [
            ("NASA released Webb's First Deep Field image showing galaxy cluster SMACS 0723 in July 2022.", "supports", "original"),
            ("Webb's First Deep Field captured galaxy cluster SMACS 0723 as it appeared 14.6 billion years ago.", "unsupported", "number"),
            ("NASA released Webb's First Deep Field image in July 2018.", "unsupported", "date"),
            ("The James Webb Space Telescope has imaged every star in the observable universe.", "unsupported", "overgeneralize"),
            ("The European Space Agency independently built and launched the Hubble successor alone.", "unsupported", "entity"),
        ]
    },
    # 7. CATL Shenxing Plus Battery
    {
        "quote": "CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range exceeding 1,000 kilometers and the ability to add 600 km of driving range in 10 minutes of ultra-fast charging.",
        "context": "Lithium iron phosphate chemistries have traditionally lagged nickel chemistries in volumetric energy density. CATL unveiled its Shenxing Plus LFP battery in April 2024, claiming a range exceeding 1,000 kilometers and the ability to add 600 km of driving range in 10 minutes of ultra-fast charging.",
        "source": "Beijing Auto Show 2024",
        "variations": [
            ("CATL's Shenxing Plus LFP battery claims to provide over 1,000 km of driving range.", "supports", "original"),
            ("The Shenxing Plus battery is claimed to recover 900 km of range in 3 minutes of charging.", "unsupported", "number"),
            ("CATL unveiled the Shenxing Plus battery in April 2020.", "unsupported", "date"),
            ("All electric vehicles in China switched to Shenxing Plus batteries by May 2024.", "unsupported", "overgeneralize"),
            ("BYD announced the Shenxing Plus battery at the Beijing Auto Show in April 2024.", "unsupported", "entity"),
        ]
    },
    # 8. TSMC 3nm Process
    {
        "quote": "TSMC commenced volume production of its 3-nanometer N3 process node in late December 2022 at Fab 18 in Southern Taiwan Science Park, offering up to 15% speed improvement at identical power compared to N5.",
        "context": "Leading-edge semiconductor manufacturing reached another threshold when TSMC commenced volume production of its 3-nanometer N3 process node in late December 2022 at Fab 18 in Southern Taiwan Science Park, offering up to 15% speed improvement at identical power compared to N5.",
        "source": "TSMC Corporate News, Dec 2022",
        "variations": [
            ("TSMC began volume production of its 3nm N3 process node in December 2022 at Fab 18.", "supports", "original"),
            ("TSMC's N3 node offers up to a 60% speed increase over N5 at identical power consumption.", "unsupported", "number"),
            ("TSMC started high-volume 3nm production in December 2018.", "unsupported", "date"),
            ("TSMC's N3 node eliminated all heat dissipation issues in modern mobile chipsets.", "unsupported", "overgeneralize"),
            ("Intel commenced volume manufacturing of its N3 node in Southern Taiwan in December 2022.", "unsupported", "entity"),
        ]
    },
    # 9. National Ignition Facility Net Energy Gain
    {
        "quote": "On December 5, 2022, the National Ignition Facility achieved fusion ignition for the first time in history, delivering 2.05 megajoules of laser energy to the target and producing 3.15 megajoules of fusion energy output.",
        "context": "Controlled fusion research achieved scientific breakeven in late 2022. On December 5, 2022, the National Ignition Facility achieved fusion ignition for the first time in history, delivering 2.05 megajoules of laser energy to the target and producing 3.15 megajoules of fusion energy output.",
        "source": "Lawrence Livermore National Laboratory (2022)",
        "variations": [
            ("The National Ignition Facility achieved fusion ignition on December 5, 2022, yielding 3.15 MJ from 2.05 MJ of laser input.", "supports", "original"),
            ("The fusion reaction at NIF produced 31.5 megajoules of energy from 2.05 MJ of input.", "unsupported", "number"),
            ("NIF first achieved fusion ignition in December 2012.", "unsupported", "date"),
            ("The 2022 NIF ignition milestone immediately made commercial fusion power universally available on the electrical grid.", "unsupported", "overgeneralize"),
            ("The ITER tokamak in France achieved inertial fusion ignition on December 5, 2022.", "unsupported", "entity"),
        ]
    },
    # 10. FlashAttention
    {
        "quote": "FlashAttention computes exact attention with fewer memory accesses by tiling the input into blocks, achieving 2-4x wall-clock speedup compared to standard PyTorch attention on modern GPUs.",
        "context": "Transformers are slow and memory-hungry on long sequences because self-attention scales quadratically with sequence length. FlashAttention computes exact attention with fewer memory accesses by tiling the input into blocks, achieving 2-4x wall-clock speedup compared to standard PyTorch attention on modern GPUs without any approximation.",
        "source": "Dao et al., NeurIPS 2022",
        "variations": [
            ("FlashAttention achieves 2-4x speedup over standard attention by minimizing High Bandwidth Memory accesses through tiling.", "supports", "original"),
            ("FlashAttention achieves a 20-40x speedup compared to standard PyTorch attention.", "unsupported", "number"),
            ("FlashAttention was published in 2016 prior to the introduction of the Transformer.", "unsupported", "date"),
            ("FlashAttention reduces the computational complexity of self-attention from quadratic to strictly linear O(N).", "unsupported", "overgeneralize"),
            ("Triton developers proved that FlashAttention relies on low-rank matrix approximation rather than exact computation.", "unsupported", "entity"),
        ]
    },
    # 11. mRNA COVID-19 Vaccine Efficacy
    {
        "quote": "In phase 3 clinical trial results published in December 2020, the BNT162b2 mRNA vaccine demonstrated 95% efficacy against COVID-19 in participants 16 years of age or older.",
        "context": "Vaccine development proceeded at unprecedented speed during 2020. In phase 3 clinical trial results published in December 2020, the BNT162b2 mRNA vaccine demonstrated 95% efficacy against COVID-19 in participants 16 years of age or older, with a favorable safety profile over a median of two months of follow-up.",
        "source": "Polack et al., NEJM 2020",
        "variations": [
            ("The BNT162b2 mRNA vaccine demonstrated 95% efficacy against COVID-19 in clinical trial participants aged 16 and older.", "supports", "original"),
            ("Trial results showed BNT162b2 had 72% efficacy in participants aged 16 and older.", "unsupported", "number"),
            ("The phase 3 clinical trial for BNT162b2 was published in December 2015.", "unsupported", "date"),
            ("The mRNA vaccine conferred 100% permanent lifelong immunity against all respiratory pathogens in all participants.", "unsupported", "overgeneralize"),
            ("AstraZeneca's adenovirus vector vaccine was designated as BNT162b2 in the NEJM study.", "unsupported", "entity"),
        ]
    },
    # 12. LIGO Gravitational Waves
    {
        "quote": "On September 14, 2015, the Advanced LIGO detectors observed a transient gravitational-wave signal GW150914 from the coalescence of two stellar-mass black holes of approximately 36 and 29 solar masses.",
        "context": "A century after Einstein predicted gravitational waves, observational evidence was obtained. On September 14, 2015, the Advanced LIGO detectors observed a transient gravitational-wave signal GW150914 from the coalescence of two stellar-mass black holes of approximately 36 and 29 solar masses, resulting in a 62 solar mass black hole.",
        "source": "Abbott et al., Phys. Rev. Lett. 2016",
        "variations": [
            ("Advanced LIGO detected the gravitational-wave signal GW150914 on September 14, 2015, produced by two merging black holes.", "supports", "original"),
            ("The merging black holes in event GW150914 had masses of 150 and 120 solar masses respectively.", "unsupported", "number"),
            ("LIGO recorded its first gravitational wave detection in September 2005.", "unsupported", "date"),
            ("Gravitational waves from GW150914 caused catastrophic seismic disturbances across planet Earth.", "unsupported", "overgeneralize"),
            ("The Fermi Gamma-ray Space Telescope discovered GW150914 without terrestrial interferometers.", "unsupported", "entity"),
        ]
    },
    # 13. DART Asteroid Deflection
    {
        "quote": "NASA's Double Asteroid Redirection Test (DART) successfully impacted the asteroid moonlet Dimorphos on September 26, 2022, shortening its orbital period around Didymos by 33 minutes.",
        "context": "Planetary defense capability was tested in space for the first time. NASA's Double Asteroid Redirection Test (DART) successfully impacted the asteroid moonlet Dimorphos on September 26, 2022, shortening its orbital period around Didymos by 33 minutes, far exceeding the minimum threshold for mission success.",
        "source": "NASA Planetary Defense Coordination Office",
        "variations": [
            ("The DART spacecraft impacted Dimorphos on September 26, 2022, altering its orbital period by 33 minutes.", "supports", "original"),
            ("The DART impact shortened Dimorphos's orbital period by 4 hours and 15 minutes.", "unsupported", "number"),
            ("The DART mission collided with the asteroid Dimorphos in September 2017.", "unsupported", "date"),
            ("The DART spacecraft completely pulverized Dimorphos, leaving no remaining debris.", "unsupported", "overgeneralize"),
            ("Roscosmos conducted the DART asteroid deflection mission independently.", "unsupported", "entity"),
        ]
    },
    # 14. LLaMA Open Foundation Models
    {
        "quote": "In February 2023, Meta AI released LLaMA, a collection of foundation language models ranging from 7B to 65B parameters, trained on trillions of publicly available tokens.",
        "context": "Access to foundational LLMs widened substantially when in February 2023, Meta AI released LLaMA, a collection of foundation language models ranging from 7B to 65B parameters, trained on trillions of publicly available tokens without relying on proprietary datasets.",
        "source": "Touvron et al., Meta AI 2023",
        "variations": [
            ("Meta AI introduced the LLaMA family of models in February 2023, with parameter sizes ranging from 7B to 65B.", "supports", "original"),
            ("The original LLaMA model collection released in February 2023 scaled up to 500B parameters.", "unsupported", "number"),
            ("Meta AI publicly released the original LLaMA 7B-65B suite in February 2019.", "unsupported", "date"),
            ("LLaMA models resolved every hallucination issue present in generative AI upon release.", "unsupported", "overgeneralize"),
            ("Google DeepMind trained and released the LLaMA weights in February 2023.", "unsupported", "entity"),
        ]
    },
    # 15. Penicillin Discovery
    {
        "quote": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London after observing that a contaminated mold culture had inhibited the growth of staphylococci bacteria.",
        "context": "The modern antibiotic era began accidentally. Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London after observing that a contaminated mold culture had inhibited the growth of staphylococci bacteria. It was later mass-produced through work by Florey and Chain.",
        "source": "British Journal of Experimental Pathology 1929",
        "variations": [
            ("Alexander Fleming identified the antibacterial properties of penicillin mold in September 1928 at St. Mary's Hospital.", "supports", "original"),
            ("Fleming discovered penicillin in a sample containing over 50,000 bacterial strains in 1928.", "unsupported", "number"),
            ("Alexander Fleming first isolated penicillin mold in September 1958.", "unsupported", "date"),
            ("Fleming's 1928 discovery immediately cured all known bacterial infections worldwide.", "unsupported", "overgeneralize"),
            ("Louis Pasteur discovered penicillin after examining bacterial cultures in London.", "unsupported", "entity"),
        ]
    },
    # 16. Voyager 1 Interstellar Space
    {
        "quote": "NASA confirmed in September 2013 that Voyager 1 had crossed the heliopause into interstellar space in August 2012, at a distance of roughly 121 astronomical units from the Sun.",
        "context": "Human exploration reached unprecedented distance from Earth. NASA confirmed in September 2013 that Voyager 1 had crossed the heliopause into interstellar space in August 2012, at a distance of roughly 121 astronomical units from the Sun, making it the first human-made object to enter interstellar space.",
        "source": "Stone et al., Science 2013",
        "variations": [
            ("Voyager 1 entered interstellar space in August 2012 at a distance of approximately 121 AU from the Sun.", "supports", "original"),
            ("Voyager 1 crossed the heliopause at a distance of 450 astronomical units from the Sun.", "unsupported", "number"),
            ("NASA confirmed Voyager 1 crossed into interstellar space back in September 1999.", "unsupported", "date"),
            ("Voyager 1 has reached the nearest star system, Alpha Centauri, and sent back close-up photos.", "unsupported", "overgeneralize"),
            ("The Soviet probe Sputnik 1 was confirmed to have crossed into interstellar space in 2012.", "unsupported", "entity"),
        ]
    },
    # 17. ASML High-NA EUV Lithography
    {
        "quote": "In December 2023, ASML shipped its first pilot High-NA EUV lithography tool (Twinscan EXE:5000), featuring a 0.55 numerical aperture compared to the 0.33 NA of standard EUV systems, to Intel in Oregon.",
        "context": "Next-generation chip scaling relies on advanced optics. In December 2023, ASML shipped its first pilot High-NA EUV lithography tool (Twinscan EXE:5000), featuring a 0.55 numerical aperture compared to the 0.33 NA of standard EUV systems, to Intel in Oregon for research and development.",
        "source": "ASML Press Release, Dec 2023",
        "variations": [
            ("ASML delivered its first 0.55 NA High-NA EUV machine to Intel in December 2023.", "supports", "original"),
            ("The High-NA EUV system delivered in December 2023 features a 0.95 numerical aperture lens.", "unsupported", "number"),
            ("ASML delivered its first commercial High-NA EUV system to Intel in December 2015.", "unsupported", "date"),
            ("Every semiconductor fabrication facility in the world installed High-NA EUV tools by January 2024.", "unsupported", "overgeneralize"),
            ("Nikon manufactured the Twinscan EXE:5000 High-NA EUV tool and delivered it to Intel.", "unsupported", "entity"),
        ]
    },
    # 18. Perovskite Solar Cells Efficiency
    {
        "quote": "In November 2023, researchers at LONGi set a certified world record power conversion efficiency of 33.9% for silicon-perovskite tandem solar cells, surpassing the theoretical limit of single-junction silicon cells.",
        "context": "Photovoltaic research has increasingly focused on tandem architectures. In November 2023, researchers at LONGi set a certified world record power conversion efficiency of 33.9% for silicon-perovskite tandem solar cells, surpassing the theoretical limit of single-junction silicon cells (29.4%).",
        "source": "NREL Solar Cell Efficiency Chart 2023",
        "variations": [
            ("Researchers at LONGi achieved a certified 33.9% efficiency record for silicon-perovskite tandem cells in November 2023.", "supports", "original"),
            ("The silicon-perovskite tandem cell achieved an efficiency of 54.2% in the November 2023 test.", "unsupported", "number"),
            ("LONGi set the 33.9% tandem cell record in November 2016.", "unsupported", "date"),
            ("Tandem perovskite cells have completely replaced all fossil fuel power plants worldwide.", "unsupported", "overgeneralize"),
            ("First Solar set the 33.9% silicon-perovskite tandem efficiency record in November 2023.", "unsupported", "entity"),
        ]
    },
    # 19. GLP-1 Receptor Agonists for Weight Loss
    {
        "quote": "In the STEP 1 trial published in March 2021, once-weekly subcutaneous semaglutide 2.4 mg accompanied by lifestyle intervention resulted in a mean weight loss of 14.9% from baseline at 68 weeks, compared to 2.4% with placebo.",
        "context": "Anti-obesity pharmacotherapy saw a major shift with incretin mimetics. In the STEP 1 trial published in March 2021, once-weekly subcutaneous semaglutide 2.4 mg accompanied by lifestyle intervention resulted in a mean weight loss of 14.9% from baseline at 68 weeks, compared to 2.4% with placebo in adults with overweight or obesity.",
        "source": "Wilding et al., NEJM 2021",
        "variations": [
            ("Participants receiving 2.4 mg weekly semaglutide in the STEP 1 trial achieved a mean weight reduction of 14.9% at 68 weeks.", "supports", "original"),
            ("Patients in the STEP 1 trial lost an average of 38.5% of their body weight on semaglutide.", "unsupported", "number"),
            ("The STEP 1 trial results for semaglutide 2.4 mg were published in March 2011.", "unsupported", "date"),
            ("Weekly semaglutide permanently cures obesity in all patients with zero side effects.", "unsupported", "overgeneralize"),
            ("The STEP 1 trial evaluated once-daily oral metformin instead of semaglutide.", "unsupported", "entity"),
        ]
    },
    # 20. CERN Higgs Boson Discovery
    {
        "quote": "On July 4, 2012, the ATLAS and CMS collaborations at CERN announced the independent observation of a new particle with a mass of approximately 125 GeV, consistent with the Standard Model Higgs boson.",
        "context": "The elusive scalar boson completing the Standard Model was confirmed at the Large Hadron Collider. On July 4, 2012, the ATLAS and CMS collaborations at CERN announced the independent observation of a new particle with a mass of approximately 125 GeV, consistent with the Standard Model Higgs boson at five-sigma significance.",
        "source": "Phys. Lett. B 716 (2012)",
        "variations": [
            ("ATLAS and CMS at CERN announced the discovery of a particle near 125 GeV consistent with the Higgs boson on July 4, 2012.", "supports", "original"),
            ("The particle discovered at CERN had an observed mass of 350 GeV.", "unsupported", "number"),
            ("CERN announced the definitive observation of the Higgs boson in July 2002.", "unsupported", "date"),
            ("The Higgs boson discovery answered all unanswered physics questions in quantum gravity and dark energy.", "unsupported", "overgeneralize"),
            ("Fermilab's Tevatron team was the sole institution to discover the Higgs boson in July 2012.", "unsupported", "entity"),
        ]
    },
    # 21. Smallpox Eradication
    {
        "quote": "In May 1980, the World Health Assembly formally declared that smallpox had been eradicated worldwide following an intensive global vaccination campaign led by the WHO.",
        "context": "One of medicine's greatest triumphs was the global eradication of variola virus. In May 1980, the World Health Assembly formally declared that smallpox had been eradicated worldwide following an intensive global vaccination campaign led by the WHO, with the last naturally occurring case in 1977.",
        "source": "WHO Global Eradication Declaration (1980)",
        "variations": [
            ("The World Health Assembly declared smallpox eradicated globally in May 1980.", "supports", "original"),
            ("Smallpox was declared eradicated in 1980 after a campaign vaccinating exactly 500 people.", "unsupported", "number"),
            ("The World Health Assembly declared smallpox eradicated in May 2010.", "unsupported", "date"),
            ("The eradication of smallpox eliminated all viral pathogens affecting human populations.", "unsupported", "overgeneralize"),
            ("The Red Cross alone declared and managed the smallpox eradication without WHO participation.", "unsupported", "entity"),
        ]
    },
    # 22. R21/Matrix-M Malaria Vaccine
    {
        "quote": "In October 2023, the WHO recommended the R21/Matrix-M malaria vaccine for the prevention of malaria in children, following phase 3 trials showing 75% efficacy over 12 months in areas of seasonal transmission.",
        "context": "Malaria remains one of the world's deadliest parasitic diseases. In October 2023, the WHO recommended the R21/Matrix-M malaria vaccine for the prevention of malaria in children, following phase 3 trials showing 75% efficacy over 12 months in areas of seasonal transmission with a four-dose schedule.",
        "source": "WHO Recommendation Announcement, Oct 2023",
        "variations": [
            ("The WHO recommended the R21/Matrix-M vaccine in October 2023 after trials demonstrated 75% efficacy over 12 months in seasonal transmission areas.", "supports", "original"),
            ("Phase 3 trials of R21/Matrix-M showed an efficacy of 28% over 12 months.", "unsupported", "number"),
            ("The WHO recommended the R21/Matrix-M malaria vaccine back in October 2011.", "unsupported", "date"),
            ("R21/Matrix-M immediately halted all malaria transmissions worldwide within 24 hours of approval.", "unsupported", "overgeneralize"),
            ("Pfizer was the developer named by the WHO as the sole creator of the R21/Matrix-M vaccine.", "unsupported", "entity"),
        ]
    },
    # 23. Nvidia H100 GPU Memory Bandwidth
    {
        "quote": "Nvidia's H100 Tensor Core GPU based on the Hopper architecture delivers up to 3.35 terabytes per second of memory bandwidth using 80GB of HBM3 memory on the SXM5 form factor.",
        "context": "Accelerated computing for generative AI demands immense memory throughput. Nvidia's H100 Tensor Core GPU based on the Hopper architecture delivers up to 3.35 terabytes per second of memory bandwidth using 80GB of HBM3 memory on the SXM5 form factor, an increase of roughly 1.6x over the predecessor A100.",
        "source": "Nvidia Hopper Architecture Whitepaper",
        "variations": [
            ("Nvidia's H100 SXM5 GPU achieves up to 3.35 TB/s of memory bandwidth with 80GB of HBM3.", "supports", "original"),
            ("The H100 SXM5 GPU features 500GB of HBM3 memory delivering 25 TB/s bandwidth.", "unsupported", "number"),
            ("Nvidia launched the Hopper H100 architecture with HBM3 memory in 2014.", "unsupported", "date"),
            ("The H100 GPU can train any frontier LLM instantly without consuming electrical power.", "unsupported", "overgeneralize"),
            ("Intel manufactures and sells the H100 Tensor Core GPU based on Hopper.", "unsupported", "entity"),
        ]
    },
    # 24. Harvard Neutral Atom Quantum Computer
    {
        "quote": "In December 2023, researchers from Harvard, QuEra Computing, MIT, and NIST demonstrated an algorithmic quantum computer executing circuits with 48 logical qubits using neutral atoms in an optical tweezer array.",
        "context": "Quantum error correction reached a significant milestone in late 2023. In December 2023, researchers from Harvard, QuEra Computing, MIT, and NIST demonstrated an algorithmic quantum computer executing circuits with 48 logical qubits using neutral atoms in an optical tweezer array.",
        "source": "Bluvstein et al., Nature 2023",
        "variations": [
            ("Researchers from Harvard, QuEra, MIT, and NIST demonstrated a neutral-atom quantum processor operating 48 logical qubits in December 2023.", "supports", "original"),
            ("The Harvard team demonstrated a system with 5,000 error-corrected logical qubits in December 2023.", "unsupported", "number"),
            ("The 48 logical qubit neutral atom experiment was performed in December 2011.", "unsupported", "date"),
            ("Neutral atom architecture proved in 2023 that all other quantum hardware modalities are fundamentally impossible.", "unsupported", "overgeneralize"),
            ("Microsoft Azure researchers independently built the 48 logical qubit hardware without academic collaborators.", "unsupported", "entity"),
        ]
    },
    # 25. CRISPR Cas9 Precision Medicine
    {
        "quote": "Jennifer Doudna and Emmanuelle Charpentier showed in 2012 that the bacterial Cas9 enzyme could be programmed with a synthetic single guide RNA to cleave specific double-stranded DNA sequences in vitro.",
        "context": "Targeted genome engineering was revolutionized by adapting bacterial adaptive immunity. Jennifer Doudna and Emmanuelle Charpentier showed in 2012 that the bacterial Cas9 enzyme could be programmed with a synthetic single guide RNA to cleave specific double-stranded DNA sequences in vitro, earning the 2020 Nobel Prize in Chemistry.",
        "source": "Jinek et al., Science 2012",
        "variations": [
            ("Doudna and Charpentier demonstrated in 2012 that Cas9 can be guided by synthetic single guide RNA to cleave target DNA sequences.", "supports", "original"),
            ("The 2012 study demonstrated programmable Cas9 cleavage targeting 10 million distinct human genes simultaneously.", "unsupported", "number"),
            ("Doudna and Charpentier first published their programmable Cas9 work in Science in 1995.", "unsupported", "date"),
            ("The 2012 paper immediately cured all human genetic diseases at the moment of publication.", "unsupported", "overgeneralize"),
            ("Francis Crick and James Watson published the 2012 discovery of Cas9 programmable cleavage.", "unsupported", "entity"),
        ]
    },
    # 26. Solar Cell Shockley-Queisser Limit
    {
        "quote": "In 1961, William Shockley and Hans-Joachim Queisser calculated that the maximum theoretical thermodynamic efficiency of a single p-n junction solar cell under standard solar spectrum is approximately 33.7% for a bandgap of 1.34 eV.",
        "context": "The fundamental physical bound of single-junction photovoltaics was formalized early in solar research. In 1961, William Shockley and Hans-Joachim Queisser calculated that the maximum theoretical thermodynamic efficiency of a single p-n junction solar cell under standard solar spectrum is approximately 33.7% for a bandgap of 1.34 eV.",
        "source": "J. Appl. Phys. 32, 510 (1961)",
        "variations": [
            ("Shockley and Queisser calculated in 1961 that a single p-n junction solar cell has a maximum theoretical efficiency of roughly 33.7%.", "supports", "original"),
            ("Shockley and Queisser established the theoretical efficiency limit of single-junction solar cells at 85.5%.", "unsupported", "number"),
            ("Shockley and Queisser published their solar cell limit analysis in 2005.", "unsupported", "date"),
            ("The Shockley-Queisser limit proves that no device can ever convert more than 33.7% of sunlight under any physical architecture.", "unsupported", "overgeneralize"),
            ("Albert Einstein and Niels Bohr calculated the Shockley-Queisser limit together in 1961.", "unsupported", "entity"),
        ]
    },
    # 27. Gravitational Wave Speed of Light
    {
        "quote": "The observation of gravitational waves from GW170817 and electromagnetic signals from GRB 170817A within 1.7 seconds of each other constrained the difference between the speed of gravity and the speed of light to between -3×10^-15 and +7×10^-16.",
        "context": "Multimessenger astronomy provided tight experimental bounds on fundamental physics. The observation of gravitational waves from GW170817 and electromagnetic signals from GRB 170817A within 1.7 seconds of each other constrained the difference between the speed of gravity and the speed of light to between -3×10^-15 and +7×10^-16 times the speed of light.",
        "source": "Abbott et al., Astrophys. J. Lett. 2017",
        "variations": [
            ("Joint observations of GW170817 and GRB 170817A constrained the speed of gravity to match the speed of light to within approximately 10^-15.", "supports", "original"),
            ("The delay between gravitational and gamma ray signals in GW170817 was measured at 48 hours and 12 minutes.", "unsupported", "number"),
            ("The multimessenger observation of GW170817 was recorded in August 1987.", "unsupported", "date"),
            ("GW170817 completely disproved all aspects of Einstein's general relativity.", "unsupported", "overgeneralize"),
            ("The Hubble Space Telescope detected GW170817 alone without interferometer networks.", "unsupported", "entity"),
        ]
    },
    # 28. Attention Quadractic Bottleneck
    {
        "quote": "Standard multi-head attention requires storing an intermediate N x N attention matrix for sequence length N, consuming O(N^2) memory and compute operations per layer.",
        "context": "While attention mechanisms effectively model arbitrary token dependencies, standard multi-head attention requires storing an intermediate N x N attention matrix for sequence length N, consuming O(N^2) memory and compute operations per layer. This quadratic dependency creates a severe computational bottleneck when scaling context windows.",
        "source": "Standard Transformer Analysis",
        "variations": [
            ("Standard multi-head attention has an O(N^2) compute and memory requirement with respect to sequence length N.", "supports", "original"),
            ("Standard multi-head attention consumes strictly O(log N) memory relative to input sequence length.", "unsupported", "entity"),
            ("The quadratic memory bottleneck of multi-head attention was only discovered in late 2024.", "unsupported", "date"),
            ("No computer hardware can ever execute sequence lengths beyond 512 tokens due to attention scaling.", "unsupported", "overgeneralize"),
            ("Standard attention computes representations in strictly O(N) linear time without memory overhead.", "unsupported", "number"),
        ]
    },
    # 29. Antibiotic Resistance WHO
    {
        "quote": "In 2017, the World Health Organization published its first list of antibiotic-resistant priority pathogens, cataloguing 12 families of bacteria that pose the greatest threat to human health.",
        "context": "Antimicrobial resistance has emerged as a major global healthcare crisis. In 2017, the World Health Organization published its first list of antibiotic-resistant priority pathogens, cataloguing 12 families of bacteria that pose the greatest threat to human health, designating carbapenem-resistant Acinetobacter and Pseudomonas as critical priorities.",
        "source": "WHO Priority Pathogens List 2017",
        "variations": [
            ("In 2017, the WHO released a priority pathogen list identifying 12 bacterial families exhibiting severe antibiotic resistance.", "supports", "original"),
            ("The 2017 WHO priority list catalogued 250 families of bacteria as critical threats.", "unsupported", "number"),
            ("The WHO published its first priority antibiotic-resistant pathogen list in 1975.", "unsupported", "date"),
            ("Antibiotic-resistant bacteria have rendered every pharmaceutical antibiotic completely ineffective worldwide.", "unsupported", "overgeneralize"),
            ("The CDC unilaterally created the 2017 international priority pathogen list without WHO involvement.", "unsupported", "entity"),
        ]
    },
    # 30. Superconducting Qubits Coherence Times
    {
        "quote": "Transmon qubit coherence times have increased from under 1 microsecond in early designs in the late 2000s to over 100 microseconds in optimized 3D and planar architectures by 2020.",
        "context": "Progress in superconducting quantum circuits has relied on continuous material and design improvements. Transmon qubit coherence times have increased from under 1 microsecond in early designs in the late 2000s to over 100 microseconds in optimized 3D and planar architectures by 2020.",
        "source": "Kjaergaard et al., Annu. Rev. Condens. Matter Phys. 2020",
        "variations": [
            ("Transmon qubit coherence times improved from less than 1 microsecond to over 100 microseconds between the late 2000s and 2020.", "supports", "original"),
            ("Transmon coherence times exceeded 5 hours in continuous operation by 2020.", "unsupported", "number"),
            ("Transmon qubits achieved 100-microsecond coherence times back in 1985.", "unsupported", "date"),
            ("Transmon coherence improvements in 2020 completely eliminated all environmental decoherence in quantum systems.", "unsupported", "overgeneralize"),
            ("Trapped-ion researchers discovered the transmon architecture in the late 2000s.", "unsupported", "entity"),
        ]
    },
    # 31. SciFact Out-of-Domain: MGMT Methylation in Glioblastoma
    {
        "quote": "Epigenetic silencing of the MGMT DNA-repair gene by promoter methylation compromises DNA repair and is associated with longer survival in glioblastoma patients treated with alkylating agents such as temozolomide.",
        "context": "MGMT promoter methylation has become a key predictive biomarker in neuro-oncology. Epigenetic silencing of the MGMT DNA-repair gene by promoter methylation compromises DNA repair and is associated with longer survival in glioblastoma patients treated with alkylating agents such as temozolomide, compared to patients with unmethylated promoters.",
        "source": "SciFact / Hegi et al., NEJM 2005",
        "variations": [
            ("MGMT promoter methylation is associated with improved survival in glioblastoma patients treated with temozolomide.", "supports", "original"),
            ("MGMT promoter methylation reduces glioblastoma patient survival by 90% across all treatment groups.", "unsupported", "number"),
            ("Epigenetic silencing of the EGFR gene rather than MGMT predicts alkylating agent response in glioblastoma.", "unsupported", "entity"),
            ("MGMT methylation guarantees a 100% cure rate for all brain tumor patients without chemotherapy.", "unsupported", "overgeneralize"),
            ("MGMT promoter methylation directly causes glioblastoma tumors to transform into benign tissue.", "unsupported", "causal"),
        ]
    },
    # 32. SciFact Out-of-Domain: Microglial Activation in Neurodegeneration
    {
        "quote": "Activated microglia release pro-inflammatory cytokines including TNF-alpha and interleukin-1 beta, which in sustained elevated concentrations contribute to synaptic dysfunction and progressive neurodegeneration in mouse models of tauopathy.",
        "context": "Innate immune signaling in the central nervous system plays dual roles in neurodegenerative disease. Activated microglia release pro-inflammatory cytokines including TNF-alpha and interleukin-1 beta, which in sustained elevated concentrations contribute to synaptic dysfunction and progressive neurodegeneration in mouse models of tauopathy.",
        "source": "SciFact / Leyns et al., Cell 2017",
        "variations": [
            ("Pro-inflammatory cytokines released by activated microglia contribute to synaptic dysfunction in tauopathy mouse models.", "supports", "original"),
            ("Microglia release exactly 400 distinct cytokine classes within 5 seconds of brain injury.", "unsupported", "number"),
            ("Astrocytes rather than microglia are the sole cellular source of TNF-alpha described in the study.", "unsupported", "entity"),
            ("Microglia activation immediately halts all forms of neurodegeneration permanently across all mammal species.", "unsupported", "overgeneralize"),
            ("Elevated TNF-alpha release by microglia directly reverses Alzheimer's plaques in human clinical trials.", "unsupported", "causal"),
        ]
    },
    # 33. SciFact Out-of-Domain: Statin Therapy and Cardiovascular Risk
    {
        "quote": "A meta-analysis of 26 randomized trials demonstrated that each 1.0 mmol/L reduction in LDL cholesterol achieved through statin therapy produces a roughly 22% proportional reduction in the 5-year incidence of major vascular events.",
        "context": "Cholesterol-lowering interventions have been extensively characterized in clinical epidemiology. A meta-analysis of 26 randomized trials demonstrated that each 1.0 mmol/L reduction in LDL cholesterol achieved through statin therapy produces a roughly 22% proportional reduction in the 5-year incidence of major vascular events, regardless of initial cholesterol baseline.",
        "source": "SciFact / CTT Collaboration, Lancet 2010",
        "variations": [
            ("Lowering LDL cholesterol by 1.0 mmol/L with statin therapy reduces the 5-year risk of major vascular events by approximately 22%.", "supports", "original"),
            ("Each 1.0 mmol/L reduction in LDL cholesterol produces an 88% proportional reduction in vascular events.", "unsupported", "number"),
            ("Beta-blocker therapy rather than statin therapy was assessed in the 26 randomized trials.", "unsupported", "entity"),
            ("Statin therapy completely eliminates all cardiovascular mortality across all human demographics forever.", "unsupported", "overgeneralize"),
            ("Taking statins guarantees that no patient will ever experience arterial plaque accumulation.", "unsupported", "causal"),
        ]
    },
    # 34. SciFact Out-of-Domain: PD-L1 Expression in Lung Cancer
    {
        "quote": "In patients with untreated advanced non-small-cell lung cancer and PD-L1 expression on at least 50% of tumor cells, pembrolizumab monotherapy significantly prolonged progression-free survival compared to platinum-based doublet chemotherapy.",
        "context": "Immune checkpoint blockade has redefined standard-of-care in thoracic oncology. In patients with untreated advanced non-small-cell lung cancer and PD-L1 expression on at least 50% of tumor cells, pembrolizumab monotherapy significantly prolonged progression-free survival compared to platinum-based doublet chemotherapy, with fewer adverse events.",
        "source": "SciFact / Reck et al., NEJM 2016",
        "variations": [
            ("Pembrolizumab significantly improved progression-free survival over chemotherapy in advanced NSCLC patients with at least 50% PD-L1 expression.", "supports", "original"),
            ("Pembrolizumab was evaluated strictly in patients with PD-L1 expression below 2% in the KEYNOTE-024 trial.", "unsupported", "number"),
            ("Methotrexate monotherapy was demonstrated to outperform chemotherapy in the 2016 lung cancer trial.", "unsupported", "entity"),
            ("Pembrolizumab completely cures 100% of lung cancer patients regardless of tumor genetics or disease stage.", "unsupported", "overgeneralize"),
            ("Pembrolizumab directly eliminates tumor cells without requiring host T-cell immune system involvement.", "unsupported", "causal"),
        ]
    },
    # 35. SciFact Out-of-Domain: Telomerase and Senescence
    {
        "quote": "Ectopic expression of the human telomerase catalytic subunit (hTERT) extends the replicative lifespan of primary human fibroblasts beyond the Hayflick limit without inducing genomic instability or malignant transformation.",
        "context": "Cellular aging in human somatic tissues is linked to progressive telomere attrition. Ectopic expression of the human telomerase catalytic subunit (hTERT) extends the replicative lifespan of primary human fibroblasts beyond the Hayflick limit without inducing genomic instability or malignant transformation in vitro.",
        "source": "SciFact / Bodnar et al., Science 1998",
        "variations": [
            ("Expressing hTERT in primary human fibroblasts bypasses the Hayflick limit to extend replicative lifespan in vitro.", "supports", "original"),
            ("hTERT expression limits human fibroblasts to exactly 3 cell divisions before apoptosis.", "unsupported", "number"),
            ("Synthetic p53 protein rather than hTERT was introduced to extend fibroblast lifespan.", "unsupported", "entity"),
            ("Telomerase activation confers biological immortality to all whole organisms without side effects.", "unsupported", "overgeneralize"),
            ("hTERT expression in fibroblasts immediately causes spontaneous malignant tumor growth in all cells.", "unsupported", "causal"),
        ]
    }
]

def generate_csv(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "quote", "claim", "context", "perturbation_type", "label", "source"])

        row_id = 1
        for item in DATA:
            quote = item["quote"]
            context = item["context"]
            source = item["source"]
            for claim, label, perturbation_type in item["variations"]:
                writer.writerow([row_id, quote, claim, context, perturbation_type, label, source])
                row_id += 1

    print(f"Generated {row_id - 1} benchmark pairs in {output_path}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "evals" / "verifier_testset.csv"
    generate_csv(out)
