/**
* This file is part of SLICT.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/slict>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* SLICT is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* SLICT is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with SLICT.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "../utility.h"

using namespace Eigen;

class PoseFactorAnalytic : public ceres::SizedCostFunction<6, 7>
{
public:
    PoseFactorAnalytic() = delete;
    PoseFactorAnalytic(const Vector3d &pbar_, const Quaternd &qbar_, double wp_ = 1.0, double wq_ = 1.0)
    : pbar(pbar_), qbar(qbar_), wp(wp_), wq(wq_)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaternd Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::AngleAxis<double> Phir(qbar.inverse() * Qi);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wp*(Pi - pbar);
        residual.block<3, 1>(3, 0) = wq*Phir.axis()*Phir.angle();
        
        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = Vector3d(wp, wp, wp).asDiagonal();
                jacobian_pose_i.block<3, 3>(3, 0) = wq *Util::SO3JrightInv(Phir);
            }
        }

        return true;
    }
    // void Check(double **parameters);

private:
    Vector3d pbar;
    Quaternd qbar;

    double wp;
    double wq;
};