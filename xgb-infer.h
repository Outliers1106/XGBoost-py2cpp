#ifndef __XGBOOSTPP_H__

#define __XGBOOSTPP_H__

#include <string>
#include <xgboost/c_api.h>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <Eigen/Eigen>
#include <iostream>

class XGBoostPP
{
public:
    typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> Matrix;
    template<typename M>
    static void vector2Matrix(M& m, const typename M::Scalar * vec, Eigen::Index const rows, Eigen::Index const cols)
    {
        m = Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(vec, rows, cols);
    }

    XGBoostPP(std::string const& path, uint64_t nlabels):
        _modelPath(path),
        _nlabels(nlabels)
    {

        if (XGBoosterCreate(NULL, 0, &_booster) == 0 &&  XGBoosterLoadModel(_booster, _modelPath.c_str()) == 0){
            //LOG HERE
        }else{
            //LOG HERE
            _booster = NULL;
        }
    }

    int predict(Matrix const& features, Matrix& result)
    {
        DMatrixHandle X;
        const float* data = features.data();
        auto const nrow = features.rows();
        auto const ncol = features.cols();

        XGDMatrixCreateFromMat(data, nrow, ncol, NAN, &X);

        const float* out;
        uint64_t l;
        auto ret = XGBoosterPredict(_booster, X, 0, 0, 0, &l, &out);
        if (ret < 0){
            // LOG HERE
            return -1;
        }

        XGDMatrixFree(X);

        if (l != nrow*_nlabels){
            //LOG HERE
            return -1;
        }

        vector2Matrix(result, out, nrow, _nlabels);
        return 0;
    }

    virtual ~XGBoostPP(){
        XGBoosterFree(_booster);
    }

private:
    std::string const _modelPath;
    BoosterHandle _booster;
    uint64_t const _nlabels;
};

#endif
