This repository accomodates the files for laser-electric waveform retrieval based on the CRIME algorithm. Details on the algorithm itself and on the experimental implementation can be found in this article:

  J. Wiese, K. Brupbacher et al., "Universal and waveform-resolving dual pulse reconstruction through interferometric strong-field ionization", Optics Express 32(27), pp. 48734-48747 (2024), https://doi.org/10.1364/OE.534553
  
The project is written purely in Python, using NumPy and SciPy libraries, and includes three different variants of the CRIME algorithm, each with individual input requirements and resulting output quality. For each of the three variants (CRIME, twinCRIME, lazyCRIME), there is an equivalent *_results.py file provided that serves a twofold purpose: it visualises the results of a running/finished waveform retrieval and it enables the user to check the processing of the input data and parameters.
