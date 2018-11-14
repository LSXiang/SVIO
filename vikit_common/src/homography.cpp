/*
 * homography.cpp
 * Adaptation of PTAM-GPL HomographyInit class.
 * https://github.com/Oxford-PTAM/PTAM-GPL
 * Licence: GPLv3
 * Copyright 2008 Isis Innovation Limited
 *
 *  Created on: Sep 2, 2012
 *      by: cforster
 */

#include <vikit/homography.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace vk {

Homography::Homography(const vector<Vector2d, aligned_allocator<Vector2d> >& _fts1,
                       const vector<Vector2d, aligned_allocator<Vector2d> >& _fts2,
                       double _error_multiplier2,
                       double _thresh_in_px) :
    thresh(_thresh_in_px),
    error_multiplier2(_error_multiplier2),
    fts_c1(_fts1),
    fts_c2(_fts2)
{
}

void Homography::calcFromPlaneParams(const Vector3d& n_c1, const Vector3d& xyz_c1)
{
    double d = n_c1.dot(xyz_c1); // normal distance from plane to KF
    H_c2_from_c1 = T_c2_from_c1.rotation_matrix() + (T_c2_from_c1.translation()*n_c1.transpose())/d;
}

void Homography::calcFromMatches()
{
    // change the Eigen Vector2d type to cv Point2f, in order that used in the cv::findHomography function
    vector<cv::Point2f> src_pts(fts_c1.size()), dst_pts(fts_c1.size());
    for(size_t i=0; i<fts_c1.size(); ++i)
    {
        src_pts[i] = cv::Point2f(fts_c1[i][0], fts_c1[i][1]);
        dst_pts[i] = cv::Point2f(fts_c2[i][0], fts_c2[i][1]);
    }

    // TODO: replace this function to remove dependency from opencv! (Can try IPEE: https://github.com/tobycollins/IPPE)
    //!< 1:input array source points  2:input array result points  3:0, cv::RANSAC, cv::LMEDS, etc. 4:max reprojection error 
    //!< in the case, the reprojection error of 2 pixels in the unit plane is 2./f_length
    cv::Mat cvH = cv::findHomography(src_pts, dst_pts, CV_RANSAC, 2./error_multiplier2);
    H_c2_from_c1(0,0) = cvH.at<double>(0,0);
    H_c2_from_c1(0,1) = cvH.at<double>(0,1);
    H_c2_from_c1(0,2) = cvH.at<double>(0,2);
    H_c2_from_c1(1,0) = cvH.at<double>(1,0);
    H_c2_from_c1(1,1) = cvH.at<double>(1,1);
    H_c2_from_c1(1,2) = cvH.at<double>(1,2);
    H_c2_from_c1(2,0) = cvH.at<double>(2,0);
    H_c2_from_c1(2,1) = cvH.at<double>(2,1);
    H_c2_from_c1(2,2) = cvH.at<double>(2,2);
}

size_t Homography::computeMatchesInliers()
{
    // compute the error that all match corners in KF1 reproject to KF2
    inliers.clear(); inliers.resize(fts_c1.size());
    size_t n_inliers = 0;
    for(size_t i=0; i<fts_c1.size(); i++)
    {
        Vector2d projected = project2d(H_c2_from_c1 * unproject2d(fts_c1[i]));
        Vector2d e = fts_c2[i] - projected;
        double e_px = error_multiplier2 * e.norm();
        inliers[i] = (e_px < thresh);   // if the reprojection error < thresh, we deem the match corner points is inlier, others is outlier
        n_inliers += inliers[i];        // set the flag of the corresponding serial number, and count the inlier number
    }
    return n_inliers;

}

bool Homography::computeSE3fromMatches()
{
    calcFromMatches();
    bool res = decompose();
    if(!res)
        return false;
    computeMatchesInliers();
    findBestDecomposition();
    T_c2_from_c1 = decompositions.front().T;
    return true;
}

/**
 * The common way of homography matrix decomposition have [Faugeras SVD-based decomposition] and [Zhang SVD-based decomposition]
 * The following code is used Faugeras SVD-based decomposition function.
 * 
 * references: Motion and structure from motion in a piecewise plannar environment
 *   https://www.researchgate.net/publication/243764888_Motion_and_Structure_from_Motion_in_a_Piecewise_Planar_Environment
 *   https://gitee.com/paopaoslam/ORB-SLAM2/raw/master/ORB-SLAM2%E6%BA%90%E7%A0%81%E8%AF%A6%E8%A7%A3.pdf  Page16~23
 *   https://blog.csdn.net/kokerf/article/details/72885435
 *
 * A = dR + t(n^T)
 * SVD: A = UΛ(V^T)
 * where,
 *     |d1      |
 * Λ = |   d2   |
 *     |      d3|
 * 
 * s = det(U)det(V) s^2 = 1,  U(U^T) = V(V^T) = I
 * Λ = (sd)(s(U^T)RV) + ((U^T)t)(((V^T)n)^T) ~= d'R' + t'(n'^T)
 * => R = sUR'(V^T), t = Ut', n = Vn', d = sd' 
 */
bool Homography::decompose()
{
    decompositions.clear();
    JacobiSVD<MatrixXd> svd(H_c2_from_c1, ComputeThinU | ComputeThinV);

    Vector3d singular_values = svd.singularValues();

    double d1 = fabs(singular_values[0]); // The paper suggests the square of these (e.g. the evalues of AAT)
    double d2 = fabs(singular_values[1]); // should be used, but this is wrong. c.f. Faugeras' book.
    double d3 = fabs(singular_values[2]);

    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV(); // VT^T

    double s = U.determinant() * V.determinant();   // s = det(U)det(V)

    double dPrime_PM = d2;

    int nCase;
    if(d1 != d2 && d2 != d3)
        nCase = 1;
    else if( d1 == d2 && d2 == d3)
        nCase = 3;
    else
        nCase = 2;

    if(nCase != 1)
    {
        printf("FATAL Homography Initialization: This motion case is not implemented or is degenerate. Try again. ");
        return false;
    }

    double x1_PM;
    double x2;
    double x3_PM;

    // All below deals with the case = 1 case.
    // Case 1 implies (d1 != d3)
    { // Eq. 12
        x1_PM = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
        x2    = 0;
        x3_PM = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
    };

    double e1[4] = {1.0,-1.0, 1.0,-1.0};
    double e3[4] = {1.0, 1.0,-1.0,-1.0};

    Vector3d np;
    HomographyDecomposition decomp;

    // Case 1, d' > 0:
    //
    // compute R' & t' (have 4 solutions)
    //      |CosTheta    0    -dSinTheta|                  |x1|
    // R' = |   0        1         0    |    t' = (d1 - d3)|x2|
    //      |SinTheta    0     dCosTheta|                  |x3|
    decomp.d = s * dPrime_PM;
    for(size_t signs=0; signs<4; signs++)
    {
        // Eq 13
        decomp.R = Matrix3d::Identity();
        double dSinTheta = (d1 - d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
        double dCosTheta = (d1 * x3_PM * x3_PM + d3 * x1_PM * x1_PM) / d2;
        decomp.R(0,0) = dCosTheta;
        decomp.R(0,2) = -dSinTheta;
        decomp.R(2,0) = dSinTheta;
        decomp.R(2,2) = dCosTheta;

        // Eq 14
        decomp.t[0] = (d1 - d3) * x1_PM * e1[signs];
        decomp.t[1] = 0.0;
        decomp.t[2] = (d1 - d3) * -x3_PM * e3[signs];

        np[0] = x1_PM * e1[signs];
        np[1] = x2;
        np[2] = x3_PM * e3[signs];
        decomp.n = V * np;

        decompositions.push_back(decomp);
    }

    // Case 1, d' < 0:
    //
    // compute R' & t' (have 4 solutions)
    //      |CosTheta    0    dSinTheta|                  |x1|
    // R' = |   0       -1        0    |    t' = (d1 + d3)|x2|
    //      |SinTheta    0   -dCosTheta|                  |x3|
    decomp.d = s * -dPrime_PM;
    for(size_t signs=0; signs<4; signs++)
    {
        // Eq 15
        decomp.R = -1 * Matrix3d::Identity();
        double dSinPhi = (d1 + d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
        double dCosPhi = (d3 * x1_PM * x1_PM - d1 * x3_PM * x3_PM) / d2;
        decomp.R(0,0) = dCosPhi;
        decomp.R(0,2) = dSinPhi;
        decomp.R(2,0) = dSinPhi;
        decomp.R(2,2) = -dCosPhi;

        // Eq 16
        decomp.t[0] = (d1 + d3) * x1_PM * e1[signs];
        decomp.t[1] = 0.0;
        decomp.t[2] = (d1 + d3) * x3_PM * e3[signs];

        np[0] = x1_PM * e1[signs];
        np[1] = x2;
        np[2] = x3_PM * e3[signs];
        decomp.n = V * np;

        decompositions.push_back(decomp);
    }

    // Save rotation and translation of the decomposition
    for(unsigned int i=0; i<decompositions.size(); i++) // the size is 4+4
    {
        Matrix3d R = s * U * decompositions[i].R * V.transpose();
        Vector3d t = U * decompositions[i].t;
        decompositions[i].T = Sophus::SE3(R, t);
    }
    return true;
}

bool operator<(const HomographyDecomposition lhs, const HomographyDecomposition rhs)
{
    return lhs.score < rhs.score;
}

void Homography::findBestDecomposition()
{
    assert(decompositions.size() == 8); // a total of eight solutions of homography decomposition
    
    // references: Motion and structure from motion in a piecewise plannar environment  <see Proposition 4>
    // 1st, the z-coordinates of the corner points in the KF1&2 should > 0
    for(size_t i=0; i<decompositions.size(); i++)
    {
        HomographyDecomposition &decom = decompositions[i];
        size_t nPositive = 0;
        for(size_t m=0; m<fts_c1.size(); m++)
        {
            if(!inliers[m])
                continue;
            const Vector2d& v2 = fts_c1[m];
            // set X2=[x2, y2, z2]^T in KF2, X1=[x1, y1, z1]^T in KF1, (alpha)X2 = AX1, and alpha = d then
            //      | x2 |   | a00 a01 a02 || x1 |         | m2 |     | a00 a01 a02 || m1 | 
            // alpha| y2 | = | a10 a11 a12 || y1 |  => d*z2| n2 | = z1| a10 a11 a12 || n1 | (where m=x/z, n=y/z)
            //      | z2 |   | a20 a21 a22 || z1 |         |  1 |     | a20 a21 a22 ||  1 |
            // so, d*z2 = z1*(a20*m1 + a21*n1 + a22)
            // therefore, z2/z1 = (a20*m1 + a21*n1 + a22)/d
            double dVisibilityTest = (H_c2_from_c1(2,0) * v2[0] + H_c2_from_c1(2,1) * v2[1] + H_c2_from_c1(2,2)) / decom.d;
            if(dVisibilityTest > 0.0)
                nPositive++;
        }
        decom.score = -nPositive;
    }

    sort(decompositions.begin(), decompositions.end());   // sorts the elements in the range in ascending order. elements are compared using operator<
    decompositions.resize(4);                             // remove 4 elements with a relatively low score

    // 2nd, the z-coordinates of the corner points in the KF1 should > 0
    for(size_t i=0; i<decompositions.size(); i++)
    {
        HomographyDecomposition &decom = decompositions[i];
        int nPositive = 0;
        for(size_t m=0; m<fts_c1.size(); m++)
        {
            if(!inliers[m])
                continue;
            Vector3d v3 = unproject2d(fts_c1[m]);
            // (n^t)X1 = d, since z1 > 0
            //               | x1 |                           | m1 |
            // | nx, ny, nz || y1 | = d,   => z1| nx, ny, nz || n1 | = d,  => z1 = d/(nx*m1 + ny*n1 + nz) > 0
            //               | z1 |                           |  1 |
            double dVisibilityTest = v3.dot(decom.n) / decom.d; //!< 1/z1
            if(dVisibilityTest > 0.0)
                nPositive++;
        };
        decom.score = -nPositive;
    }

    // leaves only two solutions among the four
    sort(decompositions.begin(), decompositions.end());
    decompositions.resize(2);

    // According to Faugeras and Lustman, ambiguity exists if the two scores are equal
    // but in practive, better to look at the ratio!
    double dRatio = (double) decompositions[1].score / (double) decompositions[0].score;

    if(dRatio < 0.9) // no ambiguity!
        decompositions.erase(decompositions.begin() + 1);
    else  // two-way ambiguity. Resolve by sampsonus score of all points.
    {
        double dErrorSquaredLimit  = thresh * thresh * 4;
        double adSampsonusScores[2];
        for(size_t i=0; i<2; i++)
        {
            Sophus::SE3 T = decompositions[i].T;
            Matrix3d Essential = T.rotation_matrix() * sqew(T.translation());
            double dSumError = 0;
            for(size_t m=0; m < fts_c1.size(); m++ )
            {
                double d = sampsonusError(fts_c1[m], Essential, fts_c2[m]);
                if(d > dErrorSquaredLimit)
                    d = dErrorSquaredLimit;
                dSumError += d;
            }
            adSampsonusScores[i] = dSumError;
        }

        if(adSampsonusScores[0] <= adSampsonusScores[1])
            decompositions.erase(decompositions.begin() + 1);
        else
            decompositions.erase(decompositions.begin());
    }
}


} /* end namespace vk */
