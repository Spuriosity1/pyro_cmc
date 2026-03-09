#pragma once

#include "cell_geometry.hpp"
#include <vec3.hpp>


struct DiamondC : public CellGeometry::UnitCellSpecifier {
    DiamondC(): UnitCellSpecifier(
            imat33_t::from_cols({8,0,0}, {0,8,0}, {0,0,8}))
    {
        const std::vector<ipos_t> point_positions = {{0, 0, 0}, {2, 2, 2}};

        const std::vector<ipos_t> link_positions = {
            {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};

        CellGeometry::PointSpec pointspec;
        CellGeometry::LinkSpec linkspec;
        CellGeometry::PlaqSpec plaqspec;


        // set up the points
        for (int fcc = 0; fcc < 4; fcc++) {
            const auto& R = R_fcc[fcc];

            for (const auto &x : point_positions) {
                pointspec.position = x + R;
                this->add_point(pointspec);
            }
        }

        // set up the links
        for (int fcc = 0; fcc < 4; fcc++) {
            const auto& R = R_fcc[fcc];

            for (const auto &x : link_positions) {
                linkspec.position = x + R;
                linkspec.boundary = {{1, x}, {-1, -x}};
                this->add_link(linkspec);
            }
        }


        for (int fcc = 0; fcc < 4; fcc++) {
            const auto& R = R_fcc[fcc];

            for (int mu=0; mu<4; mu++){
                plaqspec.position = plaq_positions[mu]+R;
                plaqspec.boundary = plaq_boundaries[mu];
                this->add_plaq(plaqspec);
            }
        
        }
    }

    private:


    static constexpr ipos_t R_fcc[4] {
        {0,0,0}, {0,4,4}, {4,0,4}, {4,4,0}
    };

	// set up the plaqs	
	static constexpr ipos_t plaq_positions[4]  = {
		{-3,-3,-3}, {-3,-1,-1}, {-1,-3,-1}, {-1,-1,-3}
		//{-1,-1,-1}, {-1, 1, 1}, { 1,-1, 1}, { 1, 1,-1}
	};

    // should be static constexpr but its fucked for some reason
	const std::vector<VectorSignPair> plaq_boundaries[4] = {
		{
			{ 1,{0, -2, 2}},    {-1, {2, -2, 0}},
			{ 1,{2, 0, -2}},    {-1, {0, 2, -2}}, 
			{ 1,{-2, 2, 0}},    {-1, {-2, 0, 2}}
		},
		{
			{ 1,{ 0, 2,-2}},    {-1,{ 2, 2, 0}},
			{ 1,{ 2, 0, 2}},    {-1,{ 0,-2, 2}},
			{ 1,{-2,-2, 0}},    {-1,{-2, 0,-2}}
		},
		{
			{ 1,{ 0,-2,-2}},	{-1,{-2,-2, 0}},
			{ 1,{-2, 0, 2}},	{-1,{ 0, 2, 2}},
			{ 1,{ 2, 2, 0}},	{-1,{ 2, 0,-2}}
		},
		{
			{ 1,{ 0, 2, 2}},	{-1,{-2, 2, 0}},
			{ 1,{-2, 0,-2}},	{-1,{ 0,-2,-2}},
			{ 1,{ 2,-2, 0}},	{-1,{ 2, 0, 2}}
		}
	};
};



namespace pyrochlore {

    using idx3_t=CellGeometry::idx3_t;
    using sl_t = CellGeometry::sl_t;

static const idx3_t pyro[4] = {
    {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}
};

//const vec3_int diamond[2] = {vec3_int(), vec3_int(2,2,2)};

static const idx3_t fcc_Dy[4] = {
    {0,0,0},
    {0,4,4},
    {4,0,4},
    {4,4,0}
};

static const idx3_t fcc_Ti[4] = {
    {4,4,4},
    {4,0,0},
    {0,4,0},
    {0,0,4}};

// Vectors from a plaquette centre to its six vertices
// first index: plaquette sublattice
// second index: enumeration
static const idx3_t plaqt[4][6] = {
    {
        { 0,-2, 2},
        { 2,-2, 0},
        { 2, 0,-2},
        { 0, 2,-2},
        {-2, 2, 0},
        {-2, 0, 2}
    },
    {
        { 0, 2,-2},
        { 2, 2, 0},
        { 2, 0, 2},
        { 0,-2, 2},
        {-2,-2, 0},
        {-2, 0,-2}
    },
    {
        { 0,-2,-2},
        {-2,-2, 0},
        {-2, 0, 2},
        { 0, 2, 2},
        { 2, 2, 0},
        { 2, 0,-2}
    },
    {
        { 0, 2, 2},
        {-2, 2, 0},
        {-2, 0,-2},
        { 0,-2,-2},
        { 2,-2, 0},
        { 2, 0, 2}
    }
};

static const std::vector<std::vector<idx3_t>> nn1_dist = {
    {{0, -2, -2}, {-2, 0, -2}, {-2, -2, 0}, {0, 2, 2}, {2, 0, 2}, {2, 2, 0}},
    {{-2, 2, 0}, {-2, 0, 2}, {0, 2, 2}, {2, -2, 0}, {2, 0, -2}, {0, -2, -2}},
    {{0, -2, 2}, {2, 0, 2}, {2, -2, 0}, {0, 2, -2}, {-2, 0, -2}, {-2, 2, 0}},
    {{2, 2, 0}, {2, 0, -2}, {0, 2, -2}, {-2, -2, 0}, {-2, 0, 2}, {0, -2, 2}}
};

static const std::vector<std::vector<idx3_t>> nn2_dist = {
    {{0, -2, -2}, {-2, 0, -2}, {-2, -2, 0}, {-2, 4, 2}, {4, -2, 2}, {2, 
   4, -2}, {-2, 2, 4}, {2, -2, 4}, {4, 2, -2}}, {{-2, 2, 0}, {-2, 0, 
   2}, {0, 2, 2}, {2, -4, 2}, {2, 2, -4}, {-2, -4, -2}, {4, -2, 
   2}, {4, 2, -2}, {-2, -2, -4}}, {{0, -2, 2}, {2, 0, 2}, {2, -2, 
   0}, {2, 4, -2}, {-4, -2, -2}, {-2, 4, 2}, {2, 
   2, -4}, {-2, -2, -4}, {-4, 2, 2}}, {{2, 2, 0}, {2, 0, -2}, {0, 
   2, -2}, {-2, -4, -2}, {-2, 2, 4}, {2, -4, 2}, {-4, -2, -2}, {-4, 2,
    2}, {2, -2, 4}}
};

// 3rd neighbours WITH spin in the middle
static const std::vector<std::vector<idx3_t>> nn3a_dist = {
    {{0, 4, 4}, {4, 0, 4}, {4, 4, 0}, {0, -4, -4}, {-4, 0, -4}, {-4, -4, 0}},
    {{4, -4, 0}, {4, 0, -4}, {0, -4, -4}, {-4, 4, 0}, {-4, 0, 4}, {0, 4, 4}},
    {{0, 4, -4}, {-4, 0, -4}, {-4, 4, 0}, {0, -4, 4}, {4, 0, 4}, {4, -4, 0}},
    {{-4, -4, 0}, {-4, 0, 4}, {0, -4, 4}, {4, 4, 0}, {4, 0, -4}, {0, 4, -4}}};

static const std::vector<std::vector<idx3_t>> nn3b_dist = {
    {{-4, 4, 0}, {4, 0, -4}, {0, -4, 4}, {4, -4, 0}, {-4, 0, 4}, {0, 4, -4}},
    {{0, -4, 4}, {-4, 0, -4}, {4, 4, 0}, {0, 4, -4}, {4, 0, 4}, {-4, -4, 0}},
    {{4, 4, 0}, {-4, 0, 4}, {0, -4, -4}, {-4, -4, 0}, {4, 0, -4}, {0, 4, 4}},
    {{0, -4, -4}, {4, 0, 4}, {-4, 4, 0}, {0, 4, 4}, {-4, 0, -4}, {4, -4, 0}}};

// Square roots of 2, 3, and 6 for normalisation
#define S2 1.414213562373095048801688724209698078569671875376948073176
#define S3 1.732050807568877293527446341505872366942805253810380628055
#define S6 2.449489742783178098197284074705891391965947480656670128432

using vec3d=vector3::vec3<double>;

static const vec3d axis[4][3] = {
    {(1. / S6) * vec3d(1, 1, -2), (1. / S2) * vec3d(-1, 1, 0),
     (1. / S3) * vec3d(1, 1, 1)},
    {(1. / S6) * vec3d(1, -1, 2), (1. / S2) * vec3d(-1, -1, 0),
     (1. / S3) * vec3d(1, -1, -1)},
    {(1. / S6) * vec3d(-1, 1, 2), (1. / S2) * vec3d(1, 1, 0),
     (1. / S3) * vec3d(-1, 1, -1)},
    {(1. / S3) * vec3d(-1, -1, -2), (1. / S2) * vec3d(1, -1, 0),
     (1. / S3) * vec3d(-1, -1, 1)}
};

} // end namespace
