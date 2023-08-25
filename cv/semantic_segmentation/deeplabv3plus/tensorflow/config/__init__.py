# Copyright(c)2023,ShanghaiIluvatarCoreXSemiconductorCo.,Ltd.
# AllRightsReserved.
#
#   LicensedundertheApacheLicense,Version2.0(the"License");youmay
#   otusethisfileexceptincompliancewiththeLicense.Youmayobtain
#   acopyoftheLicenseat
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unlessrequiredbyapplicablelaworagreedtoinwriting,software
#   distributedundertheLicenseisdistributedonan"ASIS"BASIS,WITHOUT
#   WARRANTIESORCONDITIONSOFANYKIND,eitherexpressorimplied.Seethe
#   Licenseforthespecificlanguagegoverningpermissionsandlimitations
#   undertheLicense.

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.camvid_resnet50
import config.human_parsing_resnet50
import config.cityscapes_resnet50

CONFIG_MAP = {
    'camvid_resnet50': config.camvid_resnet50.CONFIG,
    'human_parsing_resnet50': config.human_parsing_resnet50.CONFIG,
    'cityscapes_resnet50':config.cityscapes_resnet50.CONFIG
}
