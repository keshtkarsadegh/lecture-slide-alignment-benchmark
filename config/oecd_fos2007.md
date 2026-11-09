# The discipline list: OECD Fields of Science and Technology (FOS 2007)

`oecd_fos2007.csv` is the second level of the OECD's Fields of Science and Technology
classification, as revised in 2007: 42 fields in 6 major fields. The codes and names are
copied verbatim from

> OECD (2007). *Revised Field of Science and Technology (FOS) Classification in the
> Frascati Manual.* Working Party of National Experts on Science and Technology
> Indicators, DSTI/EAS/STP/NESTI(2006)19/FINAL, 26 February 2007. Paris: OECD.

Only the field names are used:
the examples the OECD lists under each field are not part of the prompt.
`scripts/assign_discipline.py` asks the LLM for exactly one `code` from this file.
