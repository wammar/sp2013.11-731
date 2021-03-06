#!/usr/bin/env python

# Copyright (c) 2006 Damien Miller <djm@mindrot.org>
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# $Id: setup.py,v 1.4 2007/05/03 23:36:36 djm Exp $

import sys
from distutils.core import setup, Extension

VERSION = "0.3"

if __name__ == '__main__':
	editdist = Extension('editdist',
		sources = ['editdist.c'])
	setup(	name = "editdist",
		version = VERSION,
		author = "Damien Miller",
		author_email = "djm@mindrot.org",
		url = "http://www.mindrot.org/py-editdist.html",
		description = "Calculate Levenshtein's edit distance",
		long_description = """\
CPython module to quickly calculate Levenshtein's edit distance between
strings.
""",
		license = "BSD",
		ext_modules = [editdist]
	     )
