This repository accomodates the files for laser-electric waveform retrieval based on the CRIME algorithm. Details on the algorithm itself and on the experimental implementation can be found in this article:

J. Wiese, K. Brupbacher et al., "Universal and waveform-resolving dual pulse reconstruction through interferometric strong-field ionization", Optics Express 32(27), pp. 48734-48747 (2024), https://doi.org/10.1364/OE.534553
  
The project is written purely in Python, using NumPy and SciPy libraries, and includes three different variants of the CRIME algorithm, each with individual input requirements and resulting output quality. For each of the three variants (CRIME, twinCRIME, lazyCRIME), there is an equivalent *_results.py file provided that serves a twofold purpose: it visualises the results of a running/finished waveform retrieval and it enables the user to check the processing of the input data and parameters. All variants of the algorithm reconstruct the waveform-resolved laser-electric fields of both pulses involved in the underlying pump-probe measurement.

The basic CRIME algorithm, as described in the above-mentioned article, allows for the dual retrieval of two different, arbitrary waveforms and requires an individual spectrum and a peak fluence value for each of the two pulses as input.

Both twinCRIME and lazyCRIME assume pulse pairs with identical waveforms (apart from a scalar spectral phase offset) and thereby offer an accelerated reconstruction process, because the underlying parameter space is roughly half as large compared to CRIME. As CRIME, twinCRIME provides waveforms with accuate absolute electric field strengths, but it only requires a single spectrum and two peak fluence values as input. lazyCRIME further eliminates the need for fluence input by introducing the peak fluence values as additional optimisation parameters. The only input, besides the pump-probe trace, is a single spectrum on arbitrary intensity scale. While we found lazyCRIME to be extremely robust in retrieving the temporal shape of the waveform [Sam's paper], the absoloute scale of the resulting electric fields carries a large uncertainty.

With inputexample.h5 we provide exemplary input data to demonstrate the feed-in of measured data. All three variants of the algorithm can be tested with this sample input. For low numbers of frequency bands (n_om <= 10), they should deliver converged waveforms within minutes when run on a regular workstation computer. For reconstruction runs with much larger frequency grids, it is advisable to use computer architecture that allows for multithreading with dozens of cores. The file CRIME_monitor.py prints a summary of all running reconstruction jobs, based on the snapshot files that contain the momentarily best solutions.

Here is a brief explanation of the various input parameters:

identifier : The name of the .h5 file containing the input data like spectra and fluence values.

species : Abbreviation of the atomic/molecular target that was strong-field-ionized in the underlying pump-probe measurement. It is used to assign the (vertical) ionisation energy (add further values to d_IE0_eV if needed).

frac : Fraction of the total fluence that should be kept when setting up the frequency grid. A value close to 1 (like 0.998) enables an efficient use of parameter space and at the same time ensures that the majority of the input spectrum will be covered by the reconstruction grid. You can play a bit with the *_results.py files (check_input=True) to see how the input spectrum is mapped onto the frequency grid for the reconstruction.

n_om0 : Desired number of frequency bands. The same number will be applied for both waveforms. This number defines the size of the parameter space (CRIME: 2*n_om0, twinCRIME: n_om0 + 2, lazyCRIME: n_om0 + 4) and thus the time needed for a converged retrieval run, in a nonlinear way. The closer the pulses are to their transform limit, the less frequency bands will be necessary to achieve a good representation in the time domain.

q_set : Lower bound for the fraction of the weaker pulse's fluence that has to be contained within the observation time window (q). The closer q_set to the q value of the Fourier-limited pulse, the more accurate will the resulting absolute electric fields be. The lower q_set, the more freedom will the algorithm have to shift around some of the pulse's fluence in time, possibly resulting in a better reconstruction of the pump-probe trace. It is advisable to start with a value close to q of the Fourier-limited pulse and then reduce that number, if the reconstruction did provide sufficient agreement with the experimental pump-probe trace.

n_co : Number of computing cores that are available for multithreading. The underlying parameter optimisation problem is being solved by means of a SciPy implementation of the differential evolutionary algorithm, which can be strongly accelerated by using many CPUs in parallel.

check_input : True/False. Assume flat spectral phases and show resulting spectra and waveforms. Turn to True to visualise the input data and the impact of frac, n_om0 and q_set. Turn to False to run the actual waveform reconstruction.

Use the respective *.results.py file with check_input=True to get an idea about the setup of the frequency grid and the resulting laser-electric fields. This option assumes flat spectral phases for both waveforms and allows you a view at the pulses at their Fourier limit. That way, you can get a feeling how the choice of frac and n_om0 influences the creation of the frequency grid, and receive an upper limit for q_set.
