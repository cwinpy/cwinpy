from pathlib import Path
from typing import Union

from pesummary.core.webpage.webpage import BOOTSTRAP, page

from .peutils import set_formats

PULSAR_HEADER_FORMATS = {
    "F0": {
        "html": r"\(f_{\rm rot}\) (Hz)",
        "ultablename": "F0ROT",
        "tooltip": "The rotation frequency of the pulsar",
        "formatter": set_formats(name="F0ROT", type="html", dp=2),
    },
    "2F0": {
        "html": r"\(f_{\rm gw}\,[2f_{\rm rot}]\) (Hz)",
        "ultablename": "F0ROT",
        "tooltip": (
            "The gravitational wave frequency assuming emission from the "
            "<i>l</i>=<i>m</i>=2 mass quadrupole (twice the rotation frequency)"
        ),
        "formatter": lambda x: set_formats(name="F0ROT", type="html", dp=2)(2 * x),
    },
    "F1": {
        "html": r"\(\dot{f}_{\rm rot}\) (Hz/s)",
        "ultablename": "F1ROT",
        "tooltip": (
            "The intrinsic first derivative of the rotation frequency "
            "(i.e., the spin-down rate)"
        ),
        "formatter": set_formats(name="F1ROT", type="html", dp=2, scinot=True),
    },
    "DIST": {
        "html": "distance (kpc)",
        "ultablename": "DIST",
        "tooltip": "The distance to the pulsar",
        "formatter": set_formats(name="DIST", type="html", dp=1),
    },
    "SDLIM": {
        "html": "\(h_0\) spin-down limit",
        "ultablename": "SDLIM",
        "tooltip": (
            "The inferred spin-down limit assuming a fiducial moment of "
            "inertia of 10<sup>38</sup> kg m<sup>2</sup>"
        ),
        "formatter": set_formats(name="SDLIM", type="html", dp=1, scinot=True),
    },
}


RESULTS_HEADER_FORMATS = {
    "H0": {
        "html": r"\(h_0^{95\%}\) upper limit",
        "htmlshort": r"\(h_0^{{95\%}}\)",
        "tooltip": (
            "Upper limit on the gravitational wave amplitude for a rigidly "
            "rotating triaxial source"
        ),
        "ultablename": "H0_{}_95%UL",
        "formatter": set_formats(name="H0", type="html", dp=1, scinot=True),
        "highlight": "most constraining amplitude upper limit",
    },
    "ELL": {
        "html": r"\(\varepsilon^{95\%}\) upper limit",
        "htmlshort": r"\(\varepsilon^{{95\%}}\)",
        "tooltip": (
            "Upper limit on the fiducial neutron start ellipticity for a "
            "rigidly rotating triaxial source: |<i>I</i><sub>xx</sub> - "
            "<i>I</i><sub>yy</sub>| / <i>I<sub>zz</sub>"
        ),
        "ultablename": "ELL_{}_95%UL",
        "formatter": set_formats(name="ELL", type="html", dp=1, scinot=True),
        "highlight": "most constraining ellipiticity upper limit",
    },
    "Q22": {
        "html": r"\(Q_{22}^{95\%}\) upper limit (kg m<sup>2</sup>)",
        "htmlshort": r"\(Q_{{22}}^{{95\%}}\) kg m<sup>2</sup>",
        "tooltip": ("Upper limit on the <i>l</i>=<i>m</i>=2\) mass quadrupole moment"),
        "ultablename": "Q22_{}_95%UL",
        "formatter": set_formats(name="Q22", type="html", dp=1, scinot=True),
    },
    "SDRAT": {
        "html": r"\(h_0^{95\%}\,/\,h_0^{\rm spin-down}\)",
        "htmlshort": r"spin-down ratio \(h_0^{95\%}\,/\,h_0^{\rm sd}\)",
        "tooltip": (
            "Ratio of the observed gravitational wave amplitude upper limit "
            "to the inferred spin-down limit assuming a fiducial moment of "
            "inertia of 10<sup>38</sup> kg m<sup>2</sup>"
        ),
        "ultablename": "SDRAT_{}_95%UL",
        "formatter": set_formats(name="SDRAT", type="html"),
        "highlight": "smallest spin-down ratio",
    },
    "C21": {
        "html": r"\(C_{21}^{95\%}\) upper limit",
        "htmlshort": r"\(C_{{21}}^{{95\%}}\)",
        "tooltip": ("Upper limit on the amplitude of the <i>l</i>=2, <i>m</i>=1 mode"),
        "ultablename": "C21_{}_95%UL",
        "formatter": set_formats(name="C21", type="html", dp=1, scinot=True),
        "highlight": "most constraining C<sub>21</sub> upper limit",
    },
    "C22": {
        "html": r"\(C_{22}^{95\%}\) upper limit",
        "htmlshort": r"\(C_{{22}}^{{95\%}}\)",
        "tooltip": ("Upper limit on the amplitude of the <i>l</i>=2, <i>m</i>=2 mode"),
        "ultablename": "C22_{}_95%UL",
        "formatter": set_formats(name="C22", type="html", dp=1, scinot=True),
        "highlight": "most constraining C<sub>22</sub> upper limit",
    },
    "SNR": {
        "html": r"Optimal signal-to-noise ratio \(\rho\)",
        "htmlshort": r"\(\rho\)",
        "tooltip": (
            "The optimal matched-filter signal-to-noise ratio of the maximum "
            "a-posteriori source parameters"
        ),
        "ultablename": "SNR_{}",
        "formatter": set_formats(name="SNR", type="html", dp=1, sf=2),
        "highlight": "largest signal-to-noise ratio",
    },
    "ODDSSVN": {
        "html": r"\(\log{}_{10} \mathcal{O}\) signal vs. noise",
        "htmlshort": r"\(\log{}_{10} \mathcal{O}_{\rm SvN}\)",
        "tooltip": (
            "The odds of the data containing a coherent signal versus the "
            "data containing purely noise"
        ),
        "ultablename": "ODDSSVN_{}",
        "formatter": set_formats(name="ODDS", type="html", dp=1, sf=2),
        "highlight": "largest signal versus noise odds",
    },
    "ODDSCVI": {
        "html": r"\(\log{}_{10} \mathcal{O}\) coherent vs. incoherent",
        "htmlshort": r"\(\log{}_{10} \mathcal{O}_{\rm CvI}\)",
        "tooltip": (
            "The odds the the data containing a coherent signal versus the "
            "data containing incoherent signals <i>or</i> noise"
        ),
        "ultablename": "ODDSCVI",
        "formatter": set_formats(name="ODDS", type="html", dp=1, sf=2),
        "highlight": (
            "largest coherent signal versus incoherent signal <i>or</i> noise " "odds"
        ),
    },
}


# CSS and Javascript (including MathJAX)
SCRIPTS_AND_CSS = """    <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js'></script>
    <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js'></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@fancyapps/ui@5.0/dist/fancybox/fancybox.umd.js"></script>
    <script>Fancybox.bind('[data-fancybox="gallery"]', { }); </script>
    <script src='../js/grab.js'></script>
    <script src='../js/modal.js'></script>
    <script src='../js/multi_dropbar.js'></script>
    <script src='../js/search.js'></script>
    <script src='../js/side_bar.js'></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fancyapps/ui@5.0/dist/fancybox/fancybox.css"/>
    <link rel="stylesheet" href="../css/navbar.css">
    <link rel="stylesheet" href="../css/font.css">
    <link rel="stylesheet" href="../css/table.css">
    <link rel="stylesheet" href="../css/image_styles.css">
    <link rel="stylesheet" href="../css/watermark.css">
    <link rel="stylesheet" href="../css/toggle.css">
"""


class CWPage(page):
    """
    A child class of :class:`pesummary.core.webpage.webpage.page` for
    generating CWInPy-specific summary pages.
    """

    def _setup_navbar(self, background_colour: str = None):
        if background_colour == "navbar-dark" or background_colour is None:
            self.add_content(
                "<nav class='navbar navbar-expand-sm navbar-dark "
                "bg-dark fixed-top'>\n"
            )
        else:
            self.add_content(
                "<nav class='navbar navbar-expand-sm fixed-top "
                "navbar-custom' style='background-color: %s'>" % (background_colour)
            )
        self.add_content(
            "<button class='navbar-toggler' type='button' "
            "data-toggle='collapse' data-target='#collapsibleNavbar'>\n",
            indent=2,
        )
        self.add_content("<span class='navbar-toggler-icon'></span>\n", indent=4)
        self.add_content("</button>\n", indent=2)

    def make_navbar(
        self,
        links: dict = None,
        search: bool = False,
        background_color: str = None,
        toggle: bool = False,
    ):
        """
        Make a navigation bar in boostrap format.

        Parameters
        ----------
        links: dict, optional
            Dictionary of nav bar names and associated links. This can be
            nested down by two levels.
        search: bool, optional
            if True, search bar will be given in navbar
        """

        self._setup_navbar(background_color)
        if links is None:
            raise Exception("Please specify links for use with navbar\n")
        self.add_content(
            "<div class='collapse navbar-collapse' id='collapsibleNavbar'>\n", indent=4
        )
        self.add_content("<ul class='navbar-nav'>\n", indent=6)

        for name, link in links.items():
            if type(link) == dict:
                # set dropdown menu
                self.add_content("<li class='nav-item'>\n", indent=8)
                self.add_content("<li class='nav-item dropdown'>\n", indent=10)
                self.add_content(
                    "<a class='nav-link dropdown-toggle' "
                    "href='#' id='navbarDropdown' role='button' "
                    "data-toggle='dropdown' aria-haspopup='true' "
                    "aria-expanded='false'>\n",
                    indent=12,
                )
                self.add_content(f"{name}\n", indent=12)
                self.add_content("</a>\n", indent=12)
                self.add_content(
                    "<ul class='dropdown-menu' aria-labelledby='dropdown1'>\n",
                    indent=12,
                )

                for ddn, ddl in link.items():
                    # set drop-down links
                    if type(ddl) == dict:
                        self.add_content(
                            "<li class='dropdown-item dropdown'>\n", indent=14
                        )
                        self.add_content(
                            f"<a class='dropdown-toggle' id='{ddn}' "
                            "data-toggle='dropdown' "
                            "aria-haspopup='true' "
                            f"aria-expanded='false'>{ddn}</a>\n",
                            indent=16,
                        )
                        self.add_content(
                            "<ul class='dropdown-menu' " f"aria-labelledby='{ddn}'>\n",
                            indent=16,
                        )

                        # final layer of nesting allowed!
                        for dddn, dddl in ddl.items():
                            self.add_content(
                                "<li class='dropdown-item' "
                                "href='#' onclick='window.location"
                                f'="{dddl}"\'><a>{dddn}</a></li>\n',
                                indent=18,
                            )

                        self.add_content("</ul>", indent=16)
                        self.add_content("</li>", indent=14)
                    else:
                        self.add_content(
                            "<li class='dropdown-item' href='#' "
                            f"onclick='window.location=\"{ddl}\"'>"
                            f"<a>{ddn}</a></li>\n",
                            indent=14,
                        )

                self.add_content("</ul>\n", indent=12)
                self.add_content("</li>\n", indent=10)
            else:
                self.add_content("<li class='nav-item'>\n", indent=8)
                self.add_content(
                    f"<a class='nav-link' href='#' onclick='window.location=\"{link}\"'>{name}</a>\n",
                    indent=10,
                )

            self.add_content("</li>\n", indent=8)

        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)

        self.add_content(
            "<div class='collapse navbar-collapse' id='collapsibleNavbar'>\n", indent=4
        )
        self.add_content(
            "<ul class='navbar-nav flex-row ml-md-auto d-none d-md-flex'"
            "style='margin-right:1em;'>\n",
            indent=6,
        )
        if toggle:
            self.add_content(
                "<div style='margin-top:0.5em; margin-right: 1em;' "
                "data-toggle='tooltip' title='Activate expert mode'>",
                indent=6,
            )
            self.add_content("<label class='switch'>", indent=8)
            self.add_content(
                "<input type='checkbox' onchange='show_expert_div()'>", indent=10
            )
            self.add_content("<span class='slider round'></span>", indent=10)
            self.add_content("</label>", indent=8)
            self.add_content("</div>", indent=6)
        self.add_content("</ul>\n", indent=6)
        self.add_content("</div>\n", indent=4)
        if search:
            self.add_content(
                "<input type='text' placeholder='search' id='search'>\n", indent=4
            )
            self.add_content(
                "<button type='submit' onclick='myFunction()'>Search</button>\n",
                indent=4,
            )
        self.add_content("</nav>\n")

    def make_results_table(
        self,
        contents: dict,
        format: str = "table-striped table-sm",
        sticky_header: bool = True,
        highlight_psrs: dict = None,
    ):
        """
        Generate a table of pulsar results in bootstrap format. This is based
        on the :meth:`pesummary.core.webpage.page.make_table` method

        Parameters
        ----------
        contents: dict
            A dictionary of results content. The dictionary should be keyed to
            each pulsar. The values should be a dictionary keyed to each result
            name, with the value being either the result (as a string or
            number) or a dictionary with key-value pairs being a detector and
            its associated result.
        format: str
            The bootstrap table format(s). The default is
            "table-striped table-sm".
        sticky_header: bool
            Set whether to have a sticky header on the table. Default is True.
        highlight_psrs: dict
            A dictionary of pulsars to highlight, keyed on the pulsar string,
            with values being text to add to a tooltip for that table row.
        """

        # generate headings
        h1 = {"&nbsp;": 1}
        l1 = list(contents.values())[0]
        h1.update({r: len(l1[r]) if isinstance(l1[r], dict) else 1 for r in l1})
        ncols = sum(h1.values())

        if sticky_header:
            # add a sticky header for the table
            self.add_content(
                "<style>\n"
                ".header-fixed > tbody > tr > td,\n"
                ".header-fixed > thead > tr > th {\n"
            )
            self.add_content(f"    width: {100 / ncols}%;\n")
            self.add_content("    float: left;\n}\n</style>")

        self.add_content(
            "<style>\na.psr {\n  text-decoration-line: none;\n}\n"
            "a.psr:hover {\n  text-decoration-line: none;\n  text-shadow: 1px 2px 2px #a1a1a1;\n}\n"
            "</style>"
        )

        self.make_div(_class="container", _style="max-width:1400px", indent=18)

        self.make_div(indent=20, _class="table-responsive")
        self.add_content(
            f"<table class='table {format}' style='max-width:1400px'>\n", indent=24
        )

        headertooltips = {
            RESULTS_HEADER_FORMATS[p]["htmlshort"]: RESULTS_HEADER_FORMATS[p]["tooltip"]
            for p in RESULTS_HEADER_FORMATS
        }
        headertooltips.update(
            {
                PULSAR_HEADER_FORMATS[p]["html"]: PULSAR_HEADER_FORMATS[p]["tooltip"]
                for p in PULSAR_HEADER_FORMATS
            }
        )

        # first row of header
        self.add_content("<thead>\n", indent=26)
        self.add_content("<tr class='table-light'>\n", indent=28)
        for i, h in enumerate(h1):
            cclass = "" if not i else "class='border-left'"

            # set result name headings
            if h in headertooltips:
                self.add_content(
                    (
                        f"<th {cclass} colspan='{h1[h]}' style='text-align:center' data-toggle='tooltip' "
                        f"data-html='true' title='{headertooltips[h]}'>{h}</th>\n"
                    ),
                    indent=30,
                )
            else:
                self.add_content(
                    f"<th {cclass} colspan='{h1[h]}' style='text-align:center'>{h}</th>\n",
                    indent=30,
                )
        self.add_content("</tr>\n", indent=28)
        self.add_content("</thead>\n", indent=26)

        # second row of header
        self.add_content("<thead>\n", indent=26)
        self.add_content("<tr class='table-light'>\n", indent=28)
        self.add_content("<th>Pulsar</th>\n", indent=30)
        for h in l1:
            # set detector name headings
            if isinstance(l1[h], dict):
                for d in l1[h]:
                    self.add_content(f"<th class='border-left'>{d}</th>\n", indent=30)
            else:
                self.add_content("<th class='border-left'>&nbsp;</th>\n", indent=30)

        self.add_content("</tr>\n", indent=28)
        self.add_content("</thead>\n", indent=26)

        # add results
        self.add_content("<tbody>\n", indent=26)

        for psr in contents:
            if isinstance(highlight_psrs, dict) and psr in highlight_psrs:
                # highlight given pulsars
                self.add_content(
                    f"<tr class='table-success' data-toggle='tooltip' data-html='true' title='{highlight_psrs[psr]}'>"
                )
            else:
                self.add_content("<tr>\n", indent=28)

            self.add_content(f"<td style='white-space:nowrap'>{psr}</td>\n", indent=30)

            for r in contents[psr]:
                value = contents[psr][r]
                if isinstance(value, dict):
                    for v in value.values():
                        if v.startswith("<b>"):
                            # highlight the border
                            self.add_content(
                                f"<td class='table-info' style='white-space:nowrap'>{v}</td>\n",
                                indent=30,
                            )
                        else:
                            self.add_content(
                                f"<td class='border-left' style='white-space:nowrap'>{v}</td>\n",
                                indent=30,
                            )
                else:
                    if value.startswith("<b>"):
                        # highlight the border
                        self.add_content(
                            f"<td class='table-info' style='white-space:nowrap'>{value}</td>\n",
                            indent=30,
                        )
                    else:
                        self.add_content(
                            f"<td class='border-left' style='white-space:nowrap'>{value}</td>\n",
                            indent=30,
                        )
            self.add_content("</tr>\n", indent=28)

        self.add_content("</tbody>\n", indent=26)
        self.add_content("</table>\n", indent=24)
        self.end_div(indent=20)
        self.end_div(indent=18)

    def insert_image(
        self,
        path: str,
        justify: str = "center",
        width: int = 850,
        fancybox: bool = True,
    ):
        """Generate an image in bootstrap format.

        Parameters
        ----------
        path: str, optional
            path to the image that you would like inserted
        justify: str, optional
            justifies the image to either the left, right or center
        """

        self.make_container()
        _id = path.split("/")[-1][:-4]
        # use fancy box if requested
        string = f"<a href='{path}' data-fancybox='gallery'>" if fancybox else ""
        string += (
            f"<img src='{path}' id='{_id}' alt='No image available' "
            f"style='align-items:center; width:{width}px; cursor: pointer'"
        )
        if justify == "center":
            string += " class='mx-auto d-block'"
        elif justify == "left":
            string = string[:-1] + " float:left;'"
        elif justify == "right":
            string = string[:-1] + " float:right;'"

        string += "></a>\n" if fancybox else ">\n"
        self.add_content(string, indent=2)
        self.end_container()

    def make_heading(
        self,
        htext: str,
        hsubtext: str = None,
        hsize: Union[str, int] = "1",
        anchor: str = None,
    ):
        """
        Make a heading "h" element in a div container.

        Parameters
        ----------
        htext: str
            The text of the heading.
        hsubtext: str
            Optional small muted text after the main heading text.
        hsize: str, int
            The size of the heading tag. Default is "1" for <h1> tags.
        anchor: str
            Optional string to use as a link anchor.
        """

        self.make_div(_class="container", _style="max-width:1400px")

        subtext = (
            "" if hsubtext is None else f"<small class='text-muted'>{hsubtext}</small>"
        )

        if anchor is None:
            # use heading text as anchor
            anchortext = "".join(
                c for c in htext.lower().replace(" ", "-") if c.isalnum()
            )
        else:
            anchortext = anchor

        self.add_content(f"<h{hsize} id='{anchortext}'>{htext} {subtext}</h{hsize}>\n")
        self.end_div()

    def close(self):
        # add data-toggle script
        self.add_content(
            "<script>$('[data-toggle=\"tooltip\"]').tooltip({ html:true }); </script>"
        )

        self.add_content("</body>\n</html>\n")  # close off page
        self.html_file.close()


def make_html(
    web_dir: Union[str, Path],
    label: str,
    suffix: str = None,
    title: str = "Summary Pages",
):
    """
    Make the initial html page. Adapted from pesummary.

    Parameters
    ----------
    web_dir: str, Path
        Path to the location where you would like the html file to be saved.
    label: str
        Label used to create page name.
    suffix: str
        Suffix to page name
    title: str, optional
        Header title of html page.
    """

    if suffix is None:
        pagename = f"{label}.html"
    else:
        pagename = f"{label}_{suffix}.html"

    htmlfile = Path(web_dir) / "html" / pagename
    with open(htmlfile, "w") as f:
        bootstrap = BOOTSTRAP.replace(
            "<title>title</title>", f"<title>{title}</title>\n<head>"
        )
        bootstrap = bootstrap.split("\n")
        bootstrap[-4] = ""
        bootstrap = [j + "\n" for j in bootstrap]
        f.writelines(bootstrap)
        scripts = SCRIPTS_AND_CSS.split("\n")
        scripts = [j + "\n" for j in scripts]
        f.writelines(scripts)

    return htmlfile


def open_html(web_dir, base_url, html_page, label):
    """
    Open html page ready so you can manipulate the contents. Adapted from
    pesummary.

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    base_url: str
        url to the location where you would like the html file to be saved
    page: str
        name of the html page that you would like to edit
    """
    try:
        if html_page[-5:] == ".html":
            html_page = html_page[:-5]
    except Exception:
        pass

    htmlfile = Path(web_dir) / f"{html_page}.html"
    f = open(htmlfile, "a")

    return CWPage(f, web_dir, base_url, label)
