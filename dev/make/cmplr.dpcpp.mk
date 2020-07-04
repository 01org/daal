# file: cmplt.dpcpp.mk
#===============================================================================
# Copyright 2012-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#++
#  DPC++ Compiler defenitions for makefile
#--

PLATs.dpcpp = lnx32e win32e

CMPLRDIRSUFF.dpcpp = _dpcpp

CORE.SERV.COMPILER.dpcpp = generic

-Zl.dpcpp =
-DEBC.dpcpp = -g

COMPILER.lnx.dpcpp = dpcpp $(if $(IA_is_ia32),-m32,-m64) -stdlib=libstdc++ -fgnu-runtime -fwrapv

link.dynamic.lnx.dpcpp = dpcpp $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.dpcpp   = $(-Q)march=nocona
mc_OPT.dpcpp   = $(-Q)march=core2
mc3_OPT.dpcpp  = $(-Q)march=nehalem
avx_OPT.dpcpp  = $(-Q)march=sandybridge
avx2_OPT.dpcpp = $(-Q)march=haswell
knl_OPT.dpcpp  = $(-Q)march=knl
skx_OPT.dpcpp  = $(-Q)march=skx
