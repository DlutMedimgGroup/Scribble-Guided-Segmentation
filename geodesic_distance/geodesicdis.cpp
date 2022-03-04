#include "geodesicdis.h"

#include <QTextCodec>
#include <QDebug>
#include <QDateTime>
#include <math.h>

GeoDisScibble::GeoDisScibble()
{

}

GeoDisScibble::~GeoDisScibble()
{
    int NumOfInOut = m_listIOProperty.size();
    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        if (crntInOut->GeoDisMap != nullptr)
        {
            delete[] crntInOut->GeoDisMap;
            crntInOut->GeoDisMap = nullptr;
        }
        if (crntInOut->PropertyMap != nullptr)
        {
            delete[] crntInOut->PropertyMap;
            crntInOut->PropertyMap = nullptr;
        }
        if (crntInOut->DirectionalMap != nullptr)
        {
            delete[]  crntInOut->DirectionalMap;
            crntInOut->DirectionalMap = nullptr;
        }
        delete crntInOut;
    }
    m_listIOProperty.clear();
    if (m_InputVolume != nullptr)
    {
        delete [] m_InputVolume;
        m_InputVolume = nullptr;
    }
    if (m_InputSeedMap != nullptr)
    {
        delete [] m_InputSeedMap;
        m_InputSeedMap = nullptr;
    }
    if (m_OutputLabelMap != nullptr)
    {
        delete [] m_OutputLabelMap;
        m_OutputLabelMap = nullptr;
    }
    if (m_OriginLabelMap != nullptr)
    {
        delete [] m_OriginLabelMap;
        m_OriginLabelMap = nullptr;
    }
    if (m_ConfidenceMap != nullptr)
    {
        delete [] m_ConfidenceMap;
        m_ConfidenceMap = nullptr;
    }
    m_corelist.clear();
}

// 设置输入图像
void GeoDisScibble::SetInputImage(py::array_t<double> input)
{
    m_InputDim[0] = input.shape()[0];
    m_InputDim[1] = input.shape()[1];
    m_InputDim[2] = input.shape()[2];
    m_area = m_InputDim[0] * m_InputDim[1];
    m_vv = m_area * m_InputDim[2];

    auto r1 = input.unchecked<3>();
    m_InputVolume = new float[m_vv];
    m_ValueRange[0] = r1(0, 0, 0);
    m_ValueRange[1] = r1(0, 0, 0);
    int index = 0;
    for (int k = 0; k < m_InputDim[2]; k++)
    {
        for (int j = 0; j < m_InputDim[1]; j++)
        {
            for (int i = 0; i < m_InputDim[0]; i++)
            {
                m_InputVolume[index] = r1(i, j, k);
                if (m_InputVolume[index] < m_ValueRange[0])
                {
                    m_ValueRange[0] = m_InputVolume[index];
                }
                if (m_InputVolume[index] > m_ValueRange[1])
                {
                    m_ValueRange[1] = m_InputVolume[index];
                }
                index++;
            }
        }
    }
    double range = m_ValueRange[1] - m_ValueRange[0];
    m_ValueRange[0] += 0.05 * range;
    m_ValueRange[1] -= 0.05 * range;

//    std::cout << "m_ValueRange: " << m_ValueRange[0] << " " << m_ValueRange[1] << std::endl;
}

// 设置输入种子点图，-1背景，前景从0开始，必须和SetPropertyMap对应。
void GeoDisScibble::SetInputSeedMap(py::array_t<int> seed_map)
{
    if (seed_map.shape()[0] != m_InputDim[0] || seed_map.shape()[1] != m_InputDim[1] || seed_map.shape()[2] != m_InputDim[2])
    {
        qDebug("error: dim of seed map is unequal to input image.");
        return;
    }
    auto r2 = seed_map.unchecked<3>();
    m_InputSeedMap = new int[m_vv];
    int index = 0;
    for (int k = 0; k < m_InputDim[2]; k++)
    {
        for (int j = 0; j < m_InputDim[1]; j++)
        {
            for (int i = 0; i < m_InputDim[0]; i++)
            {
                m_InputSeedMap[index++] = r2(i, j, k);
            }
        }
    }
}
// 设置图像参数
void GeoDisScibble::SetSpacing(const std::vector<double>& spacing)
{
    m_Spacing[0] = spacing[0];
    m_Spacing[1] = spacing[1];
    m_Spacing[2] = spacing[2];
}
void GeoDisScibble::SetOrigin(const std::vector<double>& origin)
{
    m_Origin[0] = origin[0];
    m_Origin[1] = origin[1];
    m_Origin[2] = origin[2];
}

// 设置概率输入
void GeoDisScibble::SetPropertyMap(py::array_t<double> property_map)
{
    int num_of_label = property_map.shape()[0];
    auto r1 = property_map.unchecked<4>();
    for(int label = 0; label < num_of_label; label++)
    {
        structPropertyInOut* crntInOut = new structPropertyInOut;
        crntInOut->Label = label;
        crntInOut->PropertyMap = new float[m_vv];
        int index = 0;
        for (int k = 0; k < m_InputDim[2]; k++)
        {
            for (int j = 0; j < m_InputDim[1]; j++)
            {
                for (int i = 0; i < m_InputDim[0]; i++)
                {
                    crntInOut->PropertyMap[index] = r1(label, i, j, k);
                    index++;
                }
            }
        }
        m_listIOProperty.append(crntInOut);
    }
}
// 设置epoch_0的网络输出概率图
void GeoDisScibble::SetOriginLabelMap(py::array_t<int> lablemap)
{
    auto r = lablemap.unchecked<3>();
    m_OriginLabelMap = new int[m_vv];
    int index = 0;
    for (int k = 0; k < m_InputDim[2]; k++)
    {
        for (int j = 0; j < m_InputDim[1]; j++)
        {
            for (int i = 0; i < m_InputDim[0]; i++)
            {
                m_OriginLabelMap[index++] = r(i, j, k);
            }
        }
    }
}
// 设置计算参数
void GeoDisScibble::SetProperties(double w_Distance, double w_Gradient, double w_Property, double Threshold)
{
    this->m_WDistance = w_Distance;
    this->m_WGradient = w_Gradient;
    this->m_WProperty = w_Property;
    this->m_Threshold = Threshold;
}
// 设置排序周期
void GeoDisScibble::SetSortPeriod(int period)
{
    this->m_SortPeriod = period;
}
void GeoDisScibble::DebugOn()
{
    m_debug = true;
}
void GeoDisScibble::DebugOff()
{
    m_debug = false;
}

// 运算
void GeoDisScibble::Generate()
{
    QDateTime current_date_time;
    if (m_debug)
    {
        current_date_time = QDateTime::currentDateTime();
        qDebug("%s", qPrintable("Generation started at " + current_date_time.toString("hh:mm:ss.zzz ")));
    }

    this->CheckBeforeStart();
    this->MeanSmooth();
    int NumOfInOut = m_listIOProperty.size();

//    this->m_InputLabelMap = new int[m_vv];
//    this->ArgmaxOfPropertyMap(m_InputLabelMap);

    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        this->GenerateGeoDisMap(crntInOut->Label, crntInOut->GeoDisMap, crntInOut->PropertyMap, crntInOut->DirectionalMap);
    }
    this->PostProcess();

    // Get Tough Label
    this->m_OutputLabelMap = new int[m_vv];
    this->ArgmaxOfPropertyMap(m_OutputLabelMap);
    // Get Confidence
    this->GenerateConfidenceMap();

//    this->MaskProcess();

    if (m_debug)
    {
        current_date_time = QDateTime::currentDateTime();
        qDebug("%s", qPrintable("Generation finished at " + current_date_time.toString("hh:mm:ss.zzz ")));
    }

//    delete [] m_InputLabelMap;
}

//// 得到输出
//py::array_t<float> GeoDisScibble::GetOutput()
//{
//    int outdim[4];
//    int num_of_label = m_listIOProperty.size();
//    outdim[0] = num_of_label;
//    outdim[1] = m_InputDim[0];
//    outdim[2] = m_InputDim[1];
//    outdim[3] = m_InputDim[2];
//    py::array_t<double> out = py::array_t<double>(outdim);
//    auto r3 = out.mutable_unchecked<4>();

//    for(int label = 0; label < num_of_label; label++)
//    {
//        structPropertyInOut* crntInOut = m_listIOProperty.at(label);
//        float* outmap = crntInOut->PropertyMap;
//        int index = 0;
//        for (int k = 0; k < m_InputDim[2]; k++)
//        {
//            for (int j = 0; j < m_InputDim[1]; j++)
//            {
//                for (int i = 0; i < m_InputDim[0]; i++)
//                {
//                    r3(label, i, j, k) = outmap[index];
//                    index++;
//                }
//            }
//        }
//    }
//    return out;
//}

py::array_t<int> GeoDisScibble::GetToughLabelMap()
{
    py::array_t<int> out = py::array_t<int>(m_InputDim);
    auto r3 = out.mutable_unchecked<3>();
    int index = 0;
    for (int k = 0; k < m_InputDim[2]; k++)
    {
        for (int j = 0; j < m_InputDim[1]; j++)
        {
            for (int i = 0; i < m_InputDim[0]; i++)
            {
                r3(i, j, k) = m_OutputLabelMap[index];
                index++;
            }
        }
    }
    return out;
}
py::array_t<int> GeoDisScibble::GetConfidenceMap()
{
    py::array_t<int> out = py::array_t<int>(m_InputDim);
    auto r3 = out.mutable_unchecked<3>();
    int index = 0;
    for (int k = 0; k < m_InputDim[2]; k++)
    {
        for (int j = 0; j < m_InputDim[1]; j++)
        {
            for (int i = 0; i < m_InputDim[0]; i++)
            {
                r3(i, j, k) = m_ConfidenceMap[index];
                index++;
            }
        }
    }
    return out;
}

py::array_t<float> GeoDisScibble::GetGeodesicDis()
{
    int outdim[4];
    int num_of_label = m_listIOProperty.size();
    outdim[0] = num_of_label;
    outdim[1] = m_InputDim[0];
    outdim[2] = m_InputDim[1];
    outdim[3] = m_InputDim[2];
    py::array_t<float> out = py::array_t<float>(outdim);
    auto r3 = out.mutable_unchecked<4>();

    for(int label = 0; label < num_of_label; label++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(label);
        float* outmap = crntInOut->GeoDisMap;
        int index = 0;
        for (int k = 0; k < m_InputDim[2]; k++)
        {
            for (int j = 0; j < m_InputDim[1]; j++)
            {
                for (int i = 0; i < m_InputDim[0]; i++)
                {
                    r3(label, i, j, k) = outmap[index];
                    index++;
                }
            }
        }
    }
    return out;
}

void GeoDisScibble::CheckBeforeStart()
{
    if (m_InputVolume == nullptr)
    {
        qDebug("error: InputVolume is nullptr.");
        return;
    }
    if (m_InputSeedMap == nullptr)
    {
        qDebug("error: m_InputSeedMap is nullptr.");
        return;
    }
    int NumOfInOut = m_listIOProperty.size();
    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        if (crntInOut->PropertyMap == nullptr)
        {
            QString s = "error: PropertyMap with Label " + QString::number(crntInOut->Label) +
                " is nullptr!";
            qDebug("%s", qPrintable(s));
            return;
        }
        if (crntInOut->GeoDisMap != nullptr)
        {
            QString s = "error: GeoDisMap with Label " + QString::number(crntInOut->Label) +
                " is not nullptr!";
            qDebug("%s", qPrintable(s));
            return;
        }
        crntInOut->GeoDisMap = new float[m_vv];
        crntInOut->DirectionalMap = new int[m_vv];
        for (int i = 0; i < m_vv; i++)
        {
            crntInOut->GeoDisMap[i] = -1;
            crntInOut->DirectionalMap[i] = 0;
        }
    }
}
/*
void GeoDisScibble::PostProcess()
{
    int NumOfInOut = m_listIOProperty.size();
    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        for (int index = 0; index < m_vv; index++)
        {
            if (crntInOut->GeoDisMap[index] >= 0)
            {
                crntInOut->GeoDisMap[index] = exp(crntInOut->GeoDisMap[index])-1;
                crntInOut->GeoDisMap[index] = exp(-(crntInOut->GeoDisMap[index] / m_Threshold));
            }
            else
            {
                crntInOut->GeoDisMap[index] = 0;
            }
        }
    }

//    for (int i = 0; i < NumOfInOut; i++)
//    {
//        double* in = m_listIOProperty.at(i)->GeoDisMap;
//        double* out = m_listIOProperty.at(i)->PropertyMap;
//        for (int index = 0; index < m_vv; index++)
//        {
//            out[index] = in[index];
//        }
//    }

    for (int index = 0; index < m_vv; index++)
    {
        bool mask = true;
        for (int j = 0; j < NumOfInOut; j++)
        {
            if(m_listIOProperty.at(j)->GeoDisMap[index] > 0)
            {
                mask = false;
                break;
            }
        }
        if (mask)
        {
            // 用m_listIOProperty中的标签0来存储mask信息
            m_listIOProperty.at(0)->PropertyMap[index] = -1000;
        }
    }

    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        for (int j = 0; j < NumOfInOut; j++)
        {
            structPropertyInOut* targetInOut = m_listIOProperty.at(j);
            if (i == j)
            {
                for (int index = 0; index < m_vv; index++)
                {
                    if (m_listIOProperty.at(0)->PropertyMap[index] < -800)
                    {
                        continue;
                    }
                    crntInOut->PropertyMap[index] += targetInOut->GeoDisMap[index];
                }
            }
            else
            {
                for (int index = 0; index < m_vv; index++)
                {
                    if (m_listIOProperty.at(0)->PropertyMap[index] < -800)
                    {
                        continue;
                    }
                    crntInOut->PropertyMap[index] -= targetInOut->GeoDisMap[index];
                }
            }
        }
    }

//    for (int index = 0; index < m_vv; index++)
//    {
//        if (m_listIOProperty.at(0)->PropertyMap[index] < -800)
//        {
//            for (int i = 0; i < NumOfInOut; i++)
//            {
//                m_listIOProperty.at(i)->PropertyMap[index] = -1;
//            }
//        }
//        else
//        {
//            for (int i = 0; i < NumOfInOut; i++)
//            {
//                if (m_listIOProperty.at(i)->PropertyMap[index] > 1)
//                {
//                    m_listIOProperty.at(i)->PropertyMap[index] = 1;
//                }
//                if (m_listIOProperty.at(i)->PropertyMap[index] < 0)
//                {
//                    m_listIOProperty.at(i)->PropertyMap[index] = 0;
//                }
//            }
//        }
//    }

    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        delete[] crntInOut->GeoDisMap;
        crntInOut->GeoDisMap = nullptr;
    }
}
*/
void GeoDisScibble::PostProcess()
{
    int NumOfInOut = m_listIOProperty.size();
    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        for (int index = 0; index < m_vv; index++)
        {
            if (crntInOut->GeoDisMap[index] < 0)
            {
                crntInOut->GeoDisMap[index] = 0;
            }
            else
            {
                crntInOut->GeoDisMap[index] = exp(crntInOut->GeoDisMap[index])-1;
                crntInOut->GeoDisMap[index] = exp(-(crntInOut->GeoDisMap[index] / m_Threshold));
            }
        }
    }
    for (int i = 0; i < NumOfInOut; i++)
    {
        structPropertyInOut* crntInOut = m_listIOProperty.at(i);
        for (int j = 0; j < NumOfInOut; j++)
        {
            structPropertyInOut* targetInOut = m_listIOProperty.at(j);
            if (i == j)
            {
                for (int index = 0; index < m_vv; index++)
                {
//                    if (targetInOut->GeoDisMap[index] > 0.9)
//                    {
//                        crntInOut->PropertyMap[index] += 2;
//                    }
//                    else
//                    {
//                        crntInOut->PropertyMap[index] += targetInOut->GeoDisMap[index];
//                    }
                    crntInOut->PropertyMap[index] += targetInOut->GeoDisMap[index];
                }
            }
            else
            {
                for (int index = 0; index < m_vv; index++)
                {
//                    if (targetInOut->GeoDisMap[index] > 0.9)
//                    {
//                        crntInOut->PropertyMap[index] -= 2;
//                    }
//                    else
//                    {
//                        crntInOut->PropertyMap[index] -= targetInOut->GeoDisMap[index];
//                    }
                    crntInOut->PropertyMap[index] -= targetInOut->GeoDisMap[index];
                }
            }
        }
    }
}

void GeoDisScibble::MaskTracking(int index, bool* mask, int* directionalmap)
{
    int crnt_index = index;
    mask[crnt_index] = false;
    while (directionalmap[crnt_index] != 0)
    {
        crnt_index = crnt_index + directionalmap[crnt_index];
        mask[crnt_index] = false;
    }
}

void GeoDisScibble::SortAndDeduplication(float* result_map)
{
    qSort(m_corelist.begin(), m_corelist.end(),
        [result_map](int x, int y) {return result_map[x]<result_map[y]; });     //去重前需要排序
    auto it = std::unique(m_corelist.begin(), m_corelist.end());   //去除容器内重复元素
    m_corelist.erase(it, m_corelist.end());
}
void GeoDisScibble::GenerateGeoDisMap(const int label, float* result_map, float* property_map, int* directional_map)
{
    // 0 准备
    m_corelist.clear();
    if (!m_corelist.isEmpty())
    {
        qDebug("error: m_list is not empty.");
        return;
    }

    // 1 根据种子点初始化距离图，种子点处距离为0，并将点加入队列
    for (int i = 0; i < m_vv; i++)
    {
        if (m_InputSeedMap[i] == label)
        {
            result_map[i] = 0;
            m_corelist.append(i);
        }
    }
    // 2 使用队列广度优先遍历计算距离图
    int num_of_interation = 0;
    if (m_debug)
    {
        qDebug("iteration start.");
    }
    while (!m_corelist.isEmpty())
    {
        int crntIndex = m_corelist.first();
        m_corelist.removeFirst();
        this->UpdateNeighbor(crntIndex, result_map, property_map, directional_map);
        num_of_interation++;
        if (num_of_interation % m_SortPeriod == 0)
        {
            this->SortAndDeduplication(result_map);
        }
        if (num_of_interation > m_maxIndex)
        {
            break;
        }
    }
}
void GeoDisScibble::UpdateNeighbor(const int index, float* result_map, float* property_map, int* directional_map)
{
    int z = index / m_area;
    int y = (index % m_area) / m_InputDim[0];
    int x = (index % m_area) % m_InputDim[0];
    if (x > 0)
    {
        // offset: -1
        this->UpdateTargetNeighbor(index, -1, result_map, property_map, directional_map, m_Spacing[0]);
    }
    if (x < m_InputDim[0] - 1)
    {
        // offset: 1
        this->UpdateTargetNeighbor(index, 1, result_map, property_map, directional_map, m_Spacing[0]);
    }
    if (y > 0)
    {
        // offset: -m_InputDim[0]
        this->UpdateTargetNeighbor(index, -m_InputDim[0], result_map, property_map, directional_map, m_Spacing[1]);
    }
    if (y < m_InputDim[1] - 1)
    {
        // offset: m_InputDim[0]
        this->UpdateTargetNeighbor(index, m_InputDim[0], result_map, property_map, directional_map, m_Spacing[1]);
    }
    if (z > 0)
    {
        // offset: -m_area
        this->UpdateTargetNeighbor(index, -m_area, result_map, property_map, directional_map, m_Spacing[2]);
    }
    if (z < m_InputDim[2] - 1)
    {
        // offset: m_area
        this->UpdateTargetNeighbor(index, m_area, result_map, property_map, directional_map, m_Spacing[2]);
    }
}
bool GeoDisScibble::UpdateTargetNeighbor(int index, int offset, float* result_map, float* property_map, int* directional_map, double Distance)
{
    if (m_InputVolume[index + offset] < m_ValueRange[0] || m_InputVolume[index + offset] > m_ValueRange[1])
    {
        return false;
    }

    // 测地线距离由三项组成：灰度差，距离，概率图
    double Gradinet = this->GetGradient(m_InputVolume[index], m_InputVolume[index + offset]);
    double Property = 1 - property_map[index + offset];
    double result = m_WDistance*Distance + m_WGradient*Gradinet + m_WProperty*Property + 1E-5;

    result = result_map[index] + result;
    if (result > m_Threshold)
    {
        return false;
    }

    if (result_map[index + offset] == -1 || result < result_map[index + offset])
    {
        result_map[index + offset] = result;
        m_corelist.append(index + offset);
        directional_map[index + offset] = -offset;
        return true;
    }
    return false;
}
void GeoDisScibble::MeanSmooth()
{
    double* cache = new double[m_vv];
    int index = 0;

    for (int k = 1; k < m_InputDim[2] - 1; k++)
    {
        for (int j = 1; j < m_InputDim[1] - 1; j++)
        {
            index = 1 + j * m_InputDim[0] + k * m_area;
            for (int i = 1; i < m_InputDim[0] - 1; i++)
            {
                cache[index] = MeanSmoothComponent(index);
                index++;
            }
        }
    }
    index = 0;
    for (int k = 1; k < m_InputDim[2] - 1; k++)
    {
        for (int j = 1; j < m_InputDim[1] - 1; j++)
        {
            for (int i = 1; i < m_InputDim[0] - 1; i++)
            {
                m_InputVolume[index] = cache[index];
                index++;
            }
        }
    }
    delete[] cache;
}
void GeoDisScibble::ArgmaxOfPropertyMap(int* output)
{
    int channel = m_listIOProperty.size();
    for (int i = 0; i < m_vv; i++)
    {
        int out = 0;
        for(int j = 1; j < channel; j++)
        {
            if (m_listIOProperty.at(j)->PropertyMap[i] > m_listIOProperty.at(out)->PropertyMap[i])
            {
                out = j;
            }
        }
        output[i] = out;
    }
}
void GeoDisScibble::GenerateConfidenceMap()
{
    // Generate mask, influenced area is true. And init confidence map
    int* pMask = new int[m_vv]; // -2 unfocus area; -1 influenced area; 0 edge in influenced area; >0 distence from edage
    m_ConfidenceMap = new int[m_vv];
    int NumOfInOut = m_listIOProperty.size();
    for (int i = 0; i < m_vv; i++)
    {
        float max_geodesic = 0;
        for (int j = 0; j < NumOfInOut; j++)
        {
            if (m_listIOProperty.at(j)->GeoDisMap[i] > max_geodesic)
            {
                max_geodesic = m_listIOProperty.at(j)->GeoDisMap[i];
            }
        }
        if (max_geodesic > 0.1)
        {
            // focus area
            pMask[i] = -1;
            m_ConfidenceMap[i] = 1000 * max_geodesic;
        }
        else
        {
            // unfocus area
            pMask[i] = -2;
            float max_property = 0;
            for (int j = 0; j < NumOfInOut; j++)
            {
                if (m_listIOProperty.at(j)->PropertyMap[i] > max_property)
                {
                    max_property = m_listIOProperty.at(j)->PropertyMap[i];
                }
            }
            m_ConfidenceMap[i] = m_UnfocusConfidence * max_property;
            m_OutputLabelMap[i] = m_OriginLabelMap[i];
        }
    }
    // Detect edge in influneced area
    int index = 0;
    for (int k = 1; k < m_InputDim[2]-1; k++)
    {
        for (int j = 1; j < m_InputDim[1]-1; j++)
        {
            index = 1 + j * m_InputDim[0] + k * m_area;
            for (int i = 1; i < m_InputDim[0]-1; i++)
            {
                if (pMask[index] == -2)
                {
                    index++;
                    continue;
                }
                if (m_OutputLabelMap[index] != m_OutputLabelMap[index-1] ||
                        m_OutputLabelMap[index] != m_OutputLabelMap[index+1] ||
                        m_OutputLabelMap[index] != m_OutputLabelMap[index-m_InputDim[0]] ||
                        m_OutputLabelMap[index] != m_OutputLabelMap[index+m_InputDim[0]] ||
                        m_OutputLabelMap[index] != m_OutputLabelMap[index-m_area] ||
                        m_OutputLabelMap[index] != m_OutputLabelMap[index+m_area])
                {
                    pMask[index] = 0;
                }
                index++;
            }
        }
    }
    // Generate distence map of edge
    for (int step = 0; step < 50; step++)
    {
        int index = 0;
        for (int k = 1; k < m_InputDim[2]-1; k++)
        {
            for (int j = 1; j < m_InputDim[1]-1; j++)
            {
                index = 1 + j * m_InputDim[0] + k * m_area;
                for (int i = 1; i < m_InputDim[0]-1; i++)
                {
                    if (pMask[index] == -2)
                    {
                        index++;
                        continue;
                    }
                    this->UpdateMaskFromNeighbour(pMask, index, index-1);
                    this->UpdateMaskFromNeighbour(pMask, index, index+1);
                    this->UpdateMaskFromNeighbour(pMask, index, index-m_InputDim[0]);
                    this->UpdateMaskFromNeighbour(pMask, index, index+m_InputDim[0]);
                    this->UpdateMaskFromNeighbour(pMask, index, index-m_area);
                    this->UpdateMaskFromNeighbour(pMask, index, index+m_area);
                    index++;
                }
            }
        }
    }
    // Update confidence around edge
    for(int i = 0; i < m_vv; i++)
    {
        if (pMask[i] < 0 || pMask[i] >= m_SiceOfConfidenceAroundEdge)
        {
            continue;
        }
        if (m_ConfidenceMap[i] > m_ConfidenceAroundEdge[pMask[i]])
        {
            m_ConfidenceMap[i] = m_ConfidenceAroundEdge[pMask[i]];
        }
    }
    // Set unfocus area of ConfidencesMap to negative
    for(int i = 0; i < m_vv; i++)
    {
        if (pMask[i] == -2)
        {
            m_ConfidenceMap[i] = -m_ConfidenceMap[i];
        }
    }

    delete [] pMask;
}
double GeoDisScibble::MeanSmoothComponent(int c)
{
    // x: +-1; y: +-m_InputDim[0]; z: +-m_area
    return (m_InputVolume[c + 1 + m_InputDim[0] + m_area] +
        m_InputVolume[c + 1 + m_InputDim[0]] +
        m_InputVolume[c + 1 + m_InputDim[0]- m_area] +
        m_InputVolume[c + 1 + m_area] +
        m_InputVolume[c + 1] +
        m_InputVolume[c + 1 - m_area] +
        m_InputVolume[c + 1 - m_InputDim[0] + m_area] +
        m_InputVolume[c + 1 - m_InputDim[0]] +
        m_InputVolume[c + 1 - m_InputDim[0] - m_area] +
        m_InputVolume[c + m_InputDim[0] + m_area] +
        m_InputVolume[c + m_InputDim[0]] +
        m_InputVolume[c + m_InputDim[0] - m_area] +
        m_InputVolume[c + m_area] +
        m_InputVolume[c] +
        m_InputVolume[c - m_area] +
        m_InputVolume[c - m_InputDim[0] + m_area] +
        m_InputVolume[c - m_InputDim[0]] +
        m_InputVolume[c - m_InputDim[0] - m_area] +
        m_InputVolume[c - 1 + m_InputDim[0] + m_area] +
        m_InputVolume[c - 1 + m_InputDim[0]] +
        m_InputVolume[c - 1 + m_InputDim[0] - m_area] +
        m_InputVolume[c - 1 + m_area] +
        m_InputVolume[c - 1] +
        m_InputVolume[c - 1 - m_area] +
        m_InputVolume[c - 1 - m_InputDim[0] + m_area] +
        m_InputVolume[c - 1 - m_InputDim[0]] +
        m_InputVolume[c - 1 - m_InputDim[0] - m_area]) / 27;
}

double GeoDisScibble::GetGradient(const double value_1, const double value_2)
{
    return abs(value_2 - value_1);
    // return (value_2 - value_1) * (value_2 - value_1);
}

void GeoDisScibble::UpdateMaskFromNeighbour(int* pMask, const int index, const int neibindex)
{
    if (pMask[neibindex] == -1)
    {
        return;
    }
    if (pMask[index] == -1 || pMask[index] > pMask[neibindex] + 1)
    {
        pMask[index] = pMask[neibindex] + 1;
    }
}

PYBIND11_MODULE(GeodesicDis, m) {
    py::class_<GeoDisScibble>(m, "GeoDisScibble", py::dynamic_attr())
        .def(py::init())
        .def("SetInputImage", &GeoDisScibble::SetInputImage)
        .def("SetInputSeedMap", &GeoDisScibble::SetInputSeedMap)
        .def("SetSpacing", &GeoDisScibble::SetSpacing)
        .def("SetOrigin", &GeoDisScibble::SetOrigin)
        .def("SetPropertyMap", &GeoDisScibble::SetPropertyMap)
        .def("SetOriginLabelMap", &GeoDisScibble::SetOriginLabelMap)
        .def("SetProperties", &GeoDisScibble::SetProperties)
        .def("SetSortPeriod", &GeoDisScibble::SetSortPeriod)
        .def("DebugOn", &GeoDisScibble::DebugOn)
        .def("DebugOff", &GeoDisScibble::DebugOff)
        .def("Generate", &GeoDisScibble::Generate)
        //.def("GetOutput", &GeoDisScibble::GetOutput)
        .def("GetToughLabelMap", &GeoDisScibble::GetToughLabelMap)
        .def("GetConfidenceMap", &GeoDisScibble::GetConfidenceMap)
        .def("GetGeodesicDis", &GeoDisScibble::GetGeodesicDis);
}
