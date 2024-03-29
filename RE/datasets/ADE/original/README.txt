ADE Corpus Charactersitics

The ADE corpus is distributed with three files i.e. DRUG-AE.rel and DRUG-DOSE.rel and ADE-NEG.txt

DRUG-AE.rel provides relations between drugs and adverse effects.
DRUG-DOSE.rel provides relations between drugs and dosages.
ADE-NEG.txt provides all sentences in the ADE corpus that DO NOT contain any drug-related adverse effects.

The format of DRUG-AE.rel is as follows with pipe delimiters:

Column-1: PubMed-ID
Column-2: Sentence 
Column-3: Adverse-Effect
Column-4: Begin offset of Adverse-Effect at 'document level'
Column-5: End offset of Adverse-Effect at 'document level'
Column-6: Drug
Column-7: Begin offset of Drug at 'document level'
Column-8: End offset of Drug at 'document level'


The format of DRUG-DOSE.rel is as follows with pipe delimiters:

Column-1: PubMed-ID
Column-2: Sentence 
Column-3: Dose
Column-4: Begin offset of Dose at 'document level'
Column-5: End offset of Dose at 'document level'
Column-6: Drug
Column-7: Begin offset of Drug at 'document level'
Column-8: End offset of Drug at 'document level'


Document Level Structure

During annotation, documents were used in the following format:
PubMed-ID\n\nTitle\n\nAbstract
Example:

10048291
Unaccountable severe hypercalcemia in a patient treated for hypoparathyroidism with dihydrotachysterol.
This report describes a forty-seven-year-old female patient with a complex medical history. She was suffering from an unspecified interstitial lung disease, papillary thyroid carcinoma which had been treated, hypoparathyroidism after thyroidectomy for which she was receiving dihydrotachysterol and calcium, and atrial fibrillation and congestive heart failure as a result of mitral stenosis. Shortly after mitral valve replacement she developed a severe hypercalcemia (serum calcium 5.95 mmol/l) during a febrile illness. At that time anti-tuberculous agents were also being administered for presumed tuberculosis. The possible mechanisms for this severe elevation of the calcium level are discussed. Immobilization, while Paget's bone disease was present, and perhaps enhanced activation of dihydrotachysterol by rifampicin, could have led to increased calcium-release into the circulation. Continuous supplecation of calcium and vitamin D, provoked dehydration and the mechanism of the milk-alkali syndrome also contributed to this extremely high calcium level. It is concluded that hypoparathyroid patients being treated with vitamin D and calcium should be carefully monitored in the case of an intercurrent illness or a change in medication.



The format of ADE-NEG.txt is as follows:
PubMed-ID\sNEG\sSentence
Where 'NEG' is a fixed string


If you use this corpus for any publication purposes, you are requested to cite the source article:

Gurulingappa et al., Benchmark Corpus to Support Information Extraction for Adverse Drug Effects, JBI, 2012.
http://www.sciencedirect.com/science/article/pii/S1532046412000615



