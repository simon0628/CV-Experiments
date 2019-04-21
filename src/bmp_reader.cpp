#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <boost/python.hpp>
// #include <boost/python/list.hpp>

#define ERROR NULL;
using namespace std;

typedef struct tagBmpFileHeader //文件头
{
    short padding;
    unsigned short bfType;		//标识该文件为bmp文件,判断文件是否为bmp文件，即用该值与"0x4d42"比较是否相等即可，0x4d42 = 19778
    unsigned int  bfSize;		//文件大小
    unsigned short bfReserved1;	//预保留位
    unsigned short bfReserved2;	//预保留位
    unsigned int  bfOffBits;	//图像数据区的起始位置
}BmpFileHeader;//14字节

typedef struct tagBmpInfoHeader //信息头
{
    unsigned int  biSize;	//图像数据大小
    int     biWidth;	//宽度
    int     biHeight;	//高度
    unsigned short biPlanes;//为1
    unsigned short biBitCount; //像素位数，8-灰度图；24-真彩色
    unsigned int biCompression;//压缩方式
    unsigned int biSizeImage;  //图像区数据大小
    int     biXPelsPerMeter;  //水平分辨率，像素每米
    int     biYPelsPerMeter;
    unsigned int biClrUsed;   //位图实际用到的颜色数
    unsigned short biClrImportant;//位图显示过程，重要的颜色数；0--所有都重要
}BmpInfoHeader;//40字节

typedef struct tagRGBPallete //调色板
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char alpha; //预保留位
}RGBPallete;

typedef struct tagVImage
{
    BmpFileHeader FileHeader;
    BmpInfoHeader InfoHeader;

    int width;
    int height;
    int channels;
    unsigned char* data;
    unsigned char** pixel;
}VImage;

VImage* loadBMP(const char* filename)
{
    VImage* newImage = new VImage;

    ifstream bmpFile;
    bmpFile.open(filename, ifstream::binary);
    if (!bmpFile.is_open())
    {
        cerr<<"Load file failed"<<endl;
        return ERROR;
    }


    // FileHeader
    BmpFileHeader* pFileHeader = new BmpFileHeader;

    bmpFile.read((char*)(&pFileHeader->bfType), sizeof(BmpFileHeader) - sizeof(short));

    if (pFileHeader->bfType != 0x4d42)// 0x4d42 = 19778
    {
        bmpFile.close();
        return ERROR;
    }
    memcpy(&newImage->FileHeader,pFileHeader, sizeof(BmpFileHeader));

    // InfoHeader
    BmpInfoHeader* pInfoHeader = new BmpInfoHeader;
    bmpFile.read((char*)pInfoHeader, sizeof(BmpInfoHeader));
    memcpy(&newImage->InfoHeader,pInfoHeader, sizeof(BmpInfoHeader));


    int width = 0, height = 0;
    width = pInfoHeader->biWidth;
    height = pInfoHeader->biHeight;
    width = ((width + 3) >> 2) << 2;

    if (8 == pInfoHeader->biBitCount)
    {
        unsigned int palleteSize;
        if(pInfoHeader->biClrUsed != 0)
        {
            palleteSize = pInfoHeader->biClrUsed;
        }
        else
        {
            palleteSize = pow(2, pInfoHeader->biBitCount);
        }


        RGBPallete **pRGBPallete = new RGBPallete*[palleteSize * sizeof(RGBPallete)];

        for(int i = 0 ; i < palleteSize; i++)
        {
            pRGBPallete[i] = new RGBPallete;
            bmpFile.read((char*)pRGBPallete[i], sizeof(RGBPallete));
        }

        newImage->width = width;
        newImage->height = height;
        newImage->channels = 1;


        unsigned char* tempBuf = new unsigned char[(width * height)];
        memset(tempBuf, 0, width * height);
        bmpFile.read((char*)tempBuf, width * height);

        newImage->data = new unsigned char[(width * height)];
        for (int i = 0; i < height; i++)
        {
            memcpy(newImage->data + i * width, tempBuf + (height - 1 - i) * width, width);
        }


        newImage->pixel = new unsigned char*[height];
        for(int i = 0; i < height; i++)
        {
            newImage->pixel[i] = new unsigned char[width];
            for(int j = 0; j < width; j++)
            {
                newImage->pixel[i][j] = pRGBPallete[int(newImage->data[i * width + j])]->r;
            }
        }


        delete pFileHeader;
        delete pInfoHeader;
        delete pRGBPallete[];
        delete tempBuf[];

        bmpFile.close();

        return newImage;
    }
    else if (24 == pInfoHeader->biBitCount) // 真彩色
    {
        cout<<"This is a colored picture"<<endl;
        bmpFile.close();
        return ERROR;
    }
    bmpFile.close();
    return ERROR;
}


void readBMP(const char* filename)
{
    VImage* Image = loadBMP(filename);

    ofstream fout;
    fout.open("./bmp_data.tmp");
    for(int i=0;i<Image->InfoHeader.biHeight;i++)
    {
        for(int j=0;j<Image->InfoHeader.biWidth;j++)
        {
            fout<<int(Image->pixel[i][j])<<' ';
        }
        fout<<endl;
    }
    fout.close();
}

BOOST_PYTHON_MODULE(bmp_reader)
{
    using namespace boost::python;
    def("readBMP", readBMP);
}



// int main()
// {
//     VImage* Image = loadBMP("/Users/simon/Desktop/CV第二次作业/Lena.bmp");

//     ofstream fout,fout_plot;
//     fout.open("./lena_data.dat");
//     fout_plot.open("./lena_data_plot.txt");
//     for(int i=0;i<Image->InfoHeader.biHeight;i++)
//     {
//         for(int j=0;j<Image->InfoHeader.biWidth;j++)
//         {
//             fout<<int(Image->pixel[i][j])<<' ';
//             fout_plot << (".:-=+*#%@"[(int) (Image->pixel[i][j] / 32)]);
//         }
//         fout<<endl;
//         fout_plot<<endl;
//     }
//     fout.close();
//     fout_plot.close();

//     return 0;
// }