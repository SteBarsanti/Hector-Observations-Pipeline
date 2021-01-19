#ifndef TDFXY_ERRH
#define TDFXY_ERRH 1
/* C language ERROR include file
 * Generated by /Users/drama/release/messgen/r3_16/macosx_x86_64/messgen utility
 * from input file /Users/ks/acmm/2dFutil/tdfxy_err.msg at Thu Dec 10 10:44:58 2020
 */

/* Facility number */
#define TDFXY__FACNUM 1851

/* OK Status symbol */
#define TDFXY__OK 0

/* Illegal RA */
#define TDFXY__ILLRA 0xF3B8322

/* Illegal Dec */
#define TDFXY__ILLDEC 0xF3B832A

/* ZD greater than 70 degrees */
#define TDFXY__ZDERR 0xF3B8332

/* Format string exceeds available string length (ErsSPrintf) */
#define TDFXY__SPRINTF 0xF3B833A

/* Error creating directory */
#define TDFXY__DIRCREERR 0xF3B8342

/* slaDs2tp failed - star too far from Axis. */
#define TDFXY__SLA_ERR_1 0xF3B834A

/* slaDs2tp failed - Antistar on tanget plane */
#define TDFXY__SLA_ERR_2 0xF3B8352

/* slaDs2tp failed - Antistar to far from axis */
#define TDFXY__SLA_ERR_3 0xF3B835A

/* slaDs2tp failed - Unexpected error code */
#define TDFXY__SLA_ERR_UNKN 0xF3B8362

/* Illegal Mean Julian Date (MJD) specified - object cannot be observed */
#define TDFXY__ILLMJD 0xF3B836A

/* Illegal temperature supplied to cvtInit routine - outside expected range */
#define TDFXY__ILLTEMP 0xF3B8372

/* Illegal pressure supplied to cvtInit routine - outside expected range */
#define TDFXY__ILLPRES 0xF3B837A

/* Illegal humidity supplied to cvtInit routine - outside expected range */
#define TDFXY__ILLHUMI 0xF3B8382

/* Illegal wavelength supplied to cvtInit routine - outside expected range */
#define TDFXY__ILLWAVE 0xF3B838A

/* Illegal scale - cannot have a scale of zero */
#define TDFXY__ILLSCALE 0xF3B8392

/* Illegal field plate number - must be 0,1 or 2 */
#define TDFXY__ILLFIELD 0xF3B839A

/* Failed to open FPI to Sky distortion map file */
#define TDFXY__MAP_FILE_OPEN 0xF3B83A2

/* Failed to read FPI to Sky distortion map file */
#define TDFXY__MAP_FILE_READ 0xF3B83AA

#endif
