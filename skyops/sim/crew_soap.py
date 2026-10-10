"""A legacy crew-rostering service with a SOAP-style XML interface (synthetic; Section 7.6).

Everything about it is the kind of thing an agent should never see: XML envelopes, a carrier code and a
zero-padded flight number in separate fields, dates as DDMONYY, local times as "1810L", rank codes (CP, FO,
PU, CC), staff numbers and names, and faults whose codes mean "retry" or "give up" only by convention.
The MCP facade in skyops/mcp/facade.py keeps all of it behind the boundary.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

NS = "http://schemas.xmlsoap.org/soap/envelope/"
_ROSTERS = {   # (carrier, number, DDMONYY) -> (crew rows, duty note)
    ("ZZ", "0214", "09OCT26"): ([("40213", "CP", "A. Rahman"), ("40877", "FO", "L. Moreau"), ("51200", "PU", "K. Tan"),
                                 ("51344", "CC", "J. Okafor"), ("51391", "CC", "M. Silva"), ("51402", "CC", "R. Iyer"),
                                 ("51477", "CC", "S. Novak"), ("51480", "CC", "T. Berg")], "1745L", "0215L"),
    ("ZZ", "0173", "09OCT26"): ([("40101", "CP", "D. Costa"), ("40655", "FO", "P. Lindqvist"), ("51007", "PU", "H. Kaur"),
                                 ("51115", "CC", "E. Mensah"), ("51190", "CC", "C. Duarte")], "1530L", "1805L"),
}
CALLS = {"busy_next": 0}        # tests set this to make the next N calls fault with Server.Busy


def _fault(code: str, text: str) -> str:
    return (f'<soap:Envelope xmlns:soap="{NS}"><soap:Body><soap:Fault><faultcode>{code}</faultcode>'
            f"<faultstring>{text}</faultstring></soap:Fault></soap:Body></soap:Envelope>")


def handle(request_xml: str) -> str:
    """The whole legacy API: one SOAP action, GetRoster. Returns an XML envelope (a roster or a fault)."""
    if CALLS["busy_next"] > 0:
        CALLS["busy_next"] -= 1
        return _fault("soap:Server.Busy", "ROSTER ENGINE BUSY - TRY LATER")
    try:
        body = ET.fromstring(request_xml).find(f"{{{NS}}}Body")
        req = body.find("GetRoster")
        key = (req.findtext("Carrier"), req.findtext("FltNbr"), req.findtext("DepDate"))
    except (ET.ParseError, AttributeError):
        return _fault("soap:Client.BadRequest", "MALFORMED REQUEST")
    if key not in _ROSTERS:
        return _fault("soap:Client.NoSuchFlight", f"NO ROSTER FOR {key[0]}{key[1]} {key[2]}")
    crew, report, limit = _ROSTERS[key]
    rows = "".join(f"<CrewMember><StaffNo>{s}</StaffNo><Rank>{r}</Rank><Name>{n}</Name></CrewMember>" for s, r, n in crew)
    return (f'<soap:Envelope xmlns:soap="{NS}"><soap:Body><RosterResponse><Status>OK</Status>{rows}'
            f"<ReportTime>{report}</ReportTime><DutyLimit>{limit}</DutyLimit></RosterResponse></soap:Body></soap:Envelope>")
