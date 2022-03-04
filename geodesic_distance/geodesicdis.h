#ifndef GEODESICDIS_H
#define GEODESICDIS_H

#include "GeodesicDis_global.h"
#include <QList>
#include <QVector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef struct structPropertyInOut
{
    int Label = 0;
    float* PropertyMap = nullptr;
    float* GeoDisMap = nullptr;
    int* DirectionalMap = nullptr;
}structPropertyInOut;

class GeoDisScibble
{
public:
    GeoDisScibble();
    ~GeoDisScibble();

    // 设置输入图像
    void SetInputImage(py::array_t<double> input);
    // 设置输入种子点图，-1背景，前景从0开始，必须和SetPropertyMap对应。
    void SetInputSeedMap(py::array_t<int> seed_map);
    // 设置图像参数
    void SetSpacing(const std::vector<double>& spacing);
    void SetOrigin(const std::vector<double>& origin);
    // 设置概率输入
    void SetPropertyMap(py::array_t<double> property_map);
    // 设置epoch_0的网络输出概率图
    void SetOriginLabelMap(py::array_t<int> lablemap);
    // 设置计算参数
    void SetProperties(double w_Distance, double w_Gradient, double w_Property, double Threshold);
    // 设置排序周期
    void SetSortPeriod(int period);
    void DebugOn();
    void DebugOff();
    // 运算
    void Generate();
    // 得到输出
    py::array_t<float> GetGeodesicDis();
    py::array_t<int> GetToughLabelMap();
    py::array_t<int> GetConfidenceMap();

protected:

    float* m_InputVolume = nullptr;
    int* m_InputSeedMap = nullptr;
    int* m_InputLabelMap = nullptr;
    int* m_OutputLabelMap = nullptr;
    int* m_ConfidenceMap = nullptr;
    int* m_OriginLabelMap = nullptr;
    int m_InputDim[3] = { 0, 0, 0 };
    double m_Spacing[3] = { 1, 1, 1 };
    double m_Origin[3] = { 0, 0, 0 };
    bool m_debug = false;
    const int m_SiceOfConfidenceAroundEdge = 3;
    const int m_ConfidenceAroundEdge[3] = {0, 250, 500};

    int m_vv = 0;
    int m_area = 0;

    double m_WDistance = 0;
    double m_WGradient = 1;
    double m_WProperty = 0.1;
    double m_Threshold = 5;
    int m_UnfocusConfidence = 500; // max confidence is 1000

    int m_SortPeriod = 10000;
    int m_maxIndex = 2000000;

    double m_ValueRange[2] = { 0,0 };

    QList<int> m_corelist;
    QVector<structPropertyInOut*> m_listIOProperty;

    void CheckBeforeStart();
    void PostProcess();
    void MaskTracking(int index, bool* mask, int* directionalmap);
    void SortAndDeduplication(float* result_map);
    void GenerateGeoDisMap(const int label, float* result_map, float* property_map, int* directional_map);
    void UpdateNeighbor(const int index, float* result_map, float* property_map, int* directional_map);
    bool UpdateTargetNeighbor(int index, int offset, float* result_map, float* property_map, int* directional_map, double Distance);
    void MeanSmooth();
    void ArgmaxOfPropertyMap(int* output);
    void GenerateConfidenceMap();
    inline double MeanSmoothComponent(int c);
    inline double GetGradient(const double value_1, const double value_2);
    inline void UpdateMaskFromNeighbour(int* pMask, const int index, const int neibindex);
};

#endif // GEODESICDIS_H
