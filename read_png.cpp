/* Read a PNG file. You may want to return an error code if the read 
* fails (depending upon the failure). There are two "prototypes" given 
* here - one where we are given the filename, and we need to open the 
* file, and the other where we are given an open file (possibly with 
* some or all of the magic bytes read - see comments above). 
*/
#ifdef open_file               /* prototype 1 */
void read_png(char *file_name) /* We need to open the file */
{
    png_structp png_ptr;
    png_infop info_ptr;

    //这些是读出图像需要的参数，图像高。宽、深度、图像类型、交错类型。
    unsigned int sig_read = 0;
    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
    FILE *fp;

    //打开文件，把文件指针给png_prt
    if ((fp = fopen(file_name, "rb")) == NULL)
        return (ERROR);
#else no_open_file  /* prototype 2 */
void read_png(FILE *fp, unsigned int sig_read) /* file is already open */
{
    png_structp png_ptr;
    png_infop info_ptr;
    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
#endif no_open_file /* only use one prototype! */

    /* Create and initialize the png_struct with the desired error handler 
* functions. If you want to use the default stderr and longjump method, 
* you can supply NULL for the last three parameters. We also supply the 
* the compiler header file version, so that we know if the application 
* was compiled with a compatible version of the library. REQUIRED 
*/
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                     png_voidp user_error_ptr, user_error_fn, user_warning_fn);

    if (png_ptr == NULL)
    {
        fclose(fp);
        return (ERROR);
    }

    /* Allocate/initialize the memory for image information. REQUIRED. */
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, png_infopp_NULL, png_infopp_NULL);
        return (ERROR);
    }

    /* Set error handling if you are using the setjmp/longjmp method (this is 
* the normal method of doing things with libpng). REQUIRED unless you 
* set up your own error handlers in the png_create_read_struct() earlier. 
*/

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        /* Free all of the memory associated with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
        fclose(fp);
        /* If we get here, we had a problem reading the file */
        return (ERROR);
    }

/* One of the following I/O initialization methods is REQUIRED */
#ifdef streams /* PNG file I/O method 1 */
    /* Set up the input control if you are using standard C streams */
    png_init_io(png_ptr, fp);

#else no_streams  /* PNG file I/O method 2 */
    /* If you are using replacement read functions, instead of calling 
* png_init_io() here you would call: 
*/
    png_set_read_fn(png_ptr, (void *)user_io_ptr, user_read_fn);
/* where user_io_ptr is a structure you want available to the callbacks */
#endif no_streams /* Use only one I/O method! */

    /* If we have already read some of the signature */
    png_set_sig_bytes(png_ptr, sig_read);

#ifdef hilevel
    /* 
* If you have enough memory to read in the entire image at once, 
* and you need to specify only transforms that can be controlled 
* with one of the PNG_TRANSFORM_* bits (this presently excludes 
* dithering, filling, setting background, and doing gamma 
* adjustment), then you can read the entire image (including 
* pixels) into the info structure with this call: 
*/
    png_read_png(png_ptr, info_ptr, png_transforms, png_voidp_NULL);
#else
    /* OK, you're doing it the hard way, with the lower-level functions */

    /* The call to png_read_info() gives us all of the information from the 
* PNG file before the first IDAT (image data chunk). REQUIRED 
*/
    png_read_info(png_ptr, info_ptr);

    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                 &interlace_type, int_p_NULL, int_p_NULL);

    /* Set up the data transformations you want. Note that these are all 
* optional. Only call them if you want/need them. Many of the 
* transformations only work on specific types of images, and many 
* are mutually exclusive. 
*/

    /* tell libpng to strip 16 bit/color files down to 8 bits/color */
    png_set_strip_16(png_ptr);

    /* Strip alpha bytes from the input data without combining with the 
* background (not recommended). 
*/
    png_set_strip_alpha(png_ptr);

    /* Extract multiple pixels with bit depths of 1, 2, and 4 from a single 
* byte into separate bytes (useful for paletted and grayscale images). 
*/
    png_set_packing(png_ptr);

    /* Change the order of packed pixels to least significant bit first 
* (not useful if you are using png_set_packing). */
    png_set_packswap(png_ptr);

    /* Expand paletted colors into true RGB triplets */
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_rgb(png_ptr);

    /* Expand grayscale images to the full 8 bits from 1, 2, or 4 bits/pixel */
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_gray_1_2_4_to_8(png_ptr);

    /* Expand paletted or RGB images with transparency to full alpha channels 
* so the data will be available as RGBA quartets. 
*/
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    /* Set the background color to draw transparent and alpha images over. 
* It is possible to set the red, green, and blue components directly 
* for paletted images instead of supplying a palette index. Note that 
* even if the PNG file supplies a background, you are not required to 
* use it - you should use the (solid) application background if it has one. 
*/

    png_color_16 my_background, *image_background;

    if (png_get_bKGD(png_ptr, info_ptr, &image_background))
        png_set_background(png_ptr, image_background,
                           PNG_BACKGROUND_GAMMA_FILE, 1, 1.0);
    else
        png_set_background(png_ptr, &my_background,
                           PNG_BACKGROUND_GAMMA_SCREEN, 0, 1.0);

    /* Some suggestions as to how to get a screen gamma value */

    /* Note that screen gamma is the display_exponent, which includes 
* the CRT_exponent and any correction for viewing conditions */
    if (/* We have a user-defined screen gamma value */)
    {
        screen_gamma = user - defined screen_gamma;
    }
    /* This is one way that applications share the same screen gamma value */
    else if ((gamma_str = getenv("SCREEN_GAMMA")) != NULL)
    {
        screen_gamma = atof(gamma_str);
    }
    /* If we don't have another value */
    else
    {
        screen_gamma = 2.2;        /* A good guess for a PC monitors in a dimly 
lit room */
        screen_gamma = 1.7 or 1.0; /* A good guess for Mac systems */
    }

    /* Tell libpng to handle the gamma conversion for you. The final call 
* is a good guess for PC generated images, but it should be configurable 
* by the user at run time by the user. It is strongly suggested that 
* your application support gamma correction. 
*/

    int intent;

    if (png_get_sRGB(png_ptr, info_ptr, &intent))
        png_set_gamma(png_ptr, screen_gamma, 0.45455);
    else
    {
        double image_gamma;
        if (png_get_gAMA(png_ptr, info_ptr, &image_gamma))
            png_set_gamma(png_ptr, screen_gamma, image_gamma);
        else
            png_set_gamma(png_ptr, screen_gamma, 0.45455);
    }

    /* Dither RGB files down to 8 bit palette or reduce palettes 
* to the number of colors available on your screen. 
*/
    if (color_type & PNG_COLOR_MASK_COLOR)
    {
        int num_palette;
        png_colorp palette;

        /* This reduces the image to the application supplied palette */
        if (/* we have our own palette */)
        {
            /* An array of colors to which the image should be dithered */
            png_color std_color_cube[MAX_SCREEN_COLORS];

            png_set_dither(png_ptr, std_color_cube, MAX_SCREEN_COLORS,
                           MAX_SCREEN_COLORS, png_uint_16p_NULL, 0);
        }
        /* This reduces the image to the palette supplied in the file */
        else if (png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette))
        {
            png_uint_16p histogram = NULL;

            png_get_hIST(png_ptr, info_ptr, &histogram);

            png_set_dither(png_ptr, palette, num_palette,
                           max_screen_colors, histogram, 0);
        }
    }

    /* invert monochrome files to have 0 as white and 1 as black */
    png_set_invert_mono(png_ptr);

    /* If you want to shift the pixel values from the range [0,255] or 
* [0,65535] to the original [0,7] or [0,31], or whatever range the 
* colors were originally in: 
*/
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_sBIT))
    {
        png_color_8p sig_bit;

        png_get_sBIT(png_ptr, info_ptr, &sig_bit);
        png_set_shift(png_ptr, sig_bit);
    }

    /* flip the RGB pixels to BGR (or RGBA to BGRA) */
    if (color_type & PNG_COLOR_MASK_COLOR)
        png_set_bgr(png_ptr);

    /* swap the RGBA or GA data to ARGB or AG (or BGRA to ABGR) */
    png_set_swap_alpha(png_ptr);

    /* swap bytes of 16 bit files to least significant byte first */
    png_set_swap(png_ptr);

    /* Add filler (or alpha) byte (before/after each RGB triplet) */
    png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);

    /* Turn on interlace handling. REQUIRED if you are not using 
* png_read_image(). To see how to handle interlacing passes, 
* see the png_read_row() method below: 
*/
    number_passes = png_set_interlace_handling(png_ptr);

    /* Optional call to gamma correct and add the background to the palette 
* and update info structure. REQUIRED if you are expecting libpng to 
* update the palette for you (ie you selected such a transform above). 
*/
    png_read_update_info(png_ptr, info_ptr);

    /* Allocate the memory to hold the image using the fields of info_ptr. */

    /* The easiest way to read the image: */
    png_bytep row_pointers[height];

    for (row = 0; row < height; row++)
    {
        row_pointers[row] = png_malloc(png_ptr, png_get_rowbytes(png_ptr,
                                                                 info_ptr));
    }

/* Now it's time to read the image. One of these methods is REQUIRED */
#ifdef entire /* Read the entire image in one go */
    png_read_image(png_ptr, row_pointers);

#else no_entire /* Read the image one or more scanlines at a time */
    /* The other way to read images - deal with interlacing: */

    for (pass = 0; pass < number_passes; pass++)
    {
#ifdef single   /* Read the image a single row at a time */
        for (y = 0; y < height; y++)
        {
            png_read_rows(png_ptr, &row_pointers[y], png_bytepp_NULL, 1);
        }

#else no_single   /* Read the image several rows at a time */
        for (y = 0; y < height; y += number_of_rows)
        {
#ifdef sparkle    /* Read the image using the "sparkle" effect. */
            png_read_rows(png_ptr, &row_pointers[y], png_bytepp_NULL,
                          number_of_rows);
#else no_sparkle  /* Read the image using the "rectangle" effect */
            png_read_rows(png_ptr, png_bytepp_NULL, &row_pointers[y],
                          number_of_rows);
#endif no_sparkle /* use only one of these two methods */
        }

/* if you want to display the image after every pass, do 
so here */
#endif no_single  /* use only one of these two methods */
    }
#endif no_entire  /* use only one of these two methods */

    /* read rest of file, and get additional chunks in info_ptr - REQUIRED */
    png_read_end(png_ptr, info_ptr);
#endif hilevel

    /* At this point you have read the entire image */

    /* clean up after the read, and free any memory allocated - REQUIRED */
    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);

    /* close the file */
    fclose(fp);

    /* that's it */
    return (OK);
}

/* progressively read a file */
