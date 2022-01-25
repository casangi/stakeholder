##########################################################################
##########################################################################
# tclean_base.py
#
# Copyright (C) 2018
# Associated Universities, Inc. Washington DC, USA.
#
# This script is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
# License for more details.
#
# [https://open-jira.nrao.edu/browse/CAS-12428]
#
#
##########################################################################

import abc
## Base Test Abstract class
class stakeholder_baseclass_template(abc.ABC):

    @abc.abstractmethod
    def setUp(self):
        pass
    
    @abc.abstractmethod
    def tearDown(self):
        pass

    @abc.abstractmethod
    def prepData(self, msname:list):
        pass

    @abc.abstractmethod
    def delData(self, msname:list):
        pass

    