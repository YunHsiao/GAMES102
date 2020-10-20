#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> input_points;
	std::vector<Ubpa::pointf2> point_set1;
	std::vector<Ubpa::pointf2> point_set2;
	std::vector<Ubpa::pointf2> point_set3;
	std::vector<Ubpa::pointf2> point_set4;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	unsigned int opt_fitting_methods{ 15u };
	int opt_fitting_step{ 5 };
	float opt_sigma{ 0.1f };
	int opt_max_exponent{ 3 };
	float opt_lambda{ 0.05f };
};

#include "details/CanvasData_AutoRefl.inl"
