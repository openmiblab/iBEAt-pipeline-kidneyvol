"""
Task: Volumetry and shape analysis of kidneys
"""

import kidney_unetr_1_segment
import kidney_unetr_2_display

if __name__=='__main__':

    kidney_unetr_1_segment.segment_site('Sheffield', 1)
    kidney_unetr_2_display.mosaic('Sheffield')
