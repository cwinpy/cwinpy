from pathlib import Path
from typing import Union

from pesummary.core.webpage.webpage import BOOTSTRAP, OTHER_SCRIPTS, page

# add MathJAX to HOME_SCRIPTS and OTHER_SCRIPTS
SCRIPTS_AND_CSS = f"""   <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js"></script>
{OTHER_SCRIPTS}"""


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

    def insert_image(self, path, justify="center", width=850, code=None):
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
        string = (
            f"<img src='{path}' id='{_id}' alt='No image available' "
            f"style='align-items:center; width:{width}px; cursor: pointer'"
        )
        if justify == "center":
            string += " class='mx-auto d-block'"
        elif justify == "left":
            string = string[:-1] + " float:left;'"
        elif justify == "right":
            string = string[:-1] + " float:right;'"
        if code:
            string += " onclick='{}(\"{}\")'".format(code, _id)
        string += ">\n"
        self.add_content(string, indent=2)
        self.end_container()

    def close(self):
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
