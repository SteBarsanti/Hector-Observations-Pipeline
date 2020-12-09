#ifndef AATTELMDL_ERR_MSGTH
#define AATTELMDL_ERR_MSGTH 1
/*  Message table include file
 *  Generated by /Users/drama/release/messgen/r3_16/macosx_x86_64/messgen utility
 *  from input file aatTelMdl_Err.msg at Sun Apr 28 11:49:55 2019
  */
static MessTabType MessFacTab_AATTELMDL[] = {
    { 2, "PROGERR", 
        "Apparent programming error in AAT Telescope model code" },
    { 3, "INVPAR", 
        "Invalid parameter to AAT Telescope model initialisation" },
    { 4, "SLADS2TPERR", 
        "Slalib Ds2tp call failed" },
    { 5, "CVTFAILED", 
        "Conversion failed" },
    { 6, "AIRMASS", 
        "Invalid observation air-mass" },
    {0, 0, 0 }};

static MessFacType MessFac_AATTELMDL = 
  { 1055, "AATTELMDL", MessFacTab_AATTELMDL, 0 };

#endif
