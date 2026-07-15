"""

Taken from Tao Liang's prettyformula.py
GitHub link:
    https://github.com/TaoLiang120/HECeramics/blob/main/heceramics/utils/prettyformula.py

"""

import numpy as np
from pymatgen.core.composition import Composition

class PrettyFormula:
    @staticmethod
    def extract_paratheses(s):
        matches = []
        start = 0
        end = 0
        while True:
            relative_s = s[start:].find("(")
            relative_e = s[end:].find(")")
            if relative_s == -1 or relative_e == -1:
                break
            matches.append(s[start + relative_s + 1:end + relative_e])
            start += 1 + relative_s
            end += 1 + relative_e
        return matches

    @staticmethod
    def compstr2frac_formula(instr, significant_figure=6):
        comp = Composition(instr)
        newstr = ""
        for iele in range(len(comp.elements)):
            el = comp.elements[iele]
            sym = el.symbol
            frac = comp.get_atomic_fraction(el)
            frac = round(frac, significant_figure)
            newstr += sym + str(frac)
        return newstr

    @staticmethod
    def normalize_composition(compstr, significant_figure=6):
        matches = PrettyFormula.extract_paratheses(compstr)
        if len(matches) == 0:
            outstr = PrettyFormula.compstr2frac_formula(compstr, significant_figure=significant_figure)
        else:
            outstr = compstr
            for i in range(len(matches)):
                c = matches[i]
                newc = PrettyFormula.compstr2frac_formula(c, significant_figure=significant_figure)
                outstr = outstr.replace(c, newc)
        return outstr

    @staticmethod
    def get(compstr, normalization=True, significant_figure=6):
        '''

        :param compstr: input string
        :param normalization: if parentheses present, normalize the composition inside parentheses.
                              if parentheses are already using pymatgen notation, normalization is not needed (False).
        :param significant_figure:
        :return:
        '''
        if normalization:
            outstr = PrettyFormula.normalize_composition(compstr, significant_figure=significant_figure)
        else:
            outstr = compstr
        multiplier = np.power(10.0, significant_figure - 1)
        comp = Composition(outstr)
        newstr = ""
        for iele in range(len(comp.elements)):
            el = comp.elements[iele]
            sym = el.symbol
            pct = comp.get_atomic_fraction(el) * multiplier
            pct = int(pct)
            newstr += sym + str(pct)
        comp = Composition(newstr)
        pretty_formula = comp.reduced_formula
        return newstr

    @staticmethod
    def from_weight_dict(d, significant_figure=6):
        comp = Composition.from_weight_dict(d)
        return PrettyFormula.get(comp.reduced_formula, significant_figure=significant_figure)

    def __str__(self):
        return f"Usage: get(composition_string, significant_figure=6) or get_from_weight(composition_by_weights, significant_figure=6)"

    def __repr__(self):
        return self.__str__()