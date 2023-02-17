#include "xgb-infer.h"
#include <algorithm>
#include <iostream>

int main(int argc, const char* argv[])
{
    auto xgb = XGBoostPP("./checkpoint.model", 3); //特征列有10列, label有3个, 回归任何的话，这里nlabel=1即可

    //result = [0.94018376 0.0304382  0.02937809]

    XGBoostPP::Matrix features(1, 10);
    features <<
         0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
         0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258;

    XGBoostPP::Matrix y;
    auto ret = xgb.predict(features, y);
    if (ret != 0){
        std::cout << "predict error" << std::endl;
    }

    std::cout << "intput : \n" << features << std::endl << "output(cpp): \n" << y << std::endl;
    std::cout << "output(python): \n" << " 0.94018376 0.0304382  0.02937809" << std::endl;
}
