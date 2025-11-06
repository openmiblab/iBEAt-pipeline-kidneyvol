"""
Task: Volumetry and shape analysis of kidneys
"""

import kidney_nnunet_1_segment
import kidney_nnunet_2_display

if __name__=='__main__':

    kidney_nnunet_1_segment.segment_site('Sheffield', 1)
    kidney_nnunet_2_display.mosaic('Sheffield')
