/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	// TODO  Set the number of particles
	num_particles = 1000;

	for(int i=0; i<num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particles.push_back(particle);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	
	default_random_engine gen;
	
	for(int i=0; i<num_particles; i++){
		
		if(fabs(yaw_rate)>0.001){
			particles[i].x += velocity * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) / yaw_rate;
			particles[i].y += velocity * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) / yaw_rate;
		}
		else{
			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta) * delta_t;
		}
		particles[i].theta += yaw_rate * delta_t;

		normal_distribution<double> dist_x(particles[i].x,std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y,std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta,std_pos[2]);
		
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);

		particles[i].x += noise_x;
		particles[i].y += noise_y;
		particles[i].theta += noise_theta;
	}



}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i< observations.size(); i++){
		double min_dist = dist(predicted[0].x,predicted[0].y,observations[i].x,observations[i].y);
		for(int j = 1; j < predicted.size(); j++){
			double distance = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
			if(min_dist > distance){
				min_dist = distance;
				observations[i].id = predicted[j].id;
			}

		}
	}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	weights.clear();
	for(int i=0; i<num_particles;i++){

		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		std::vector<LandmarkObs> observations_in_map;
		for(int j=0; j<observations.size(); j++){
			
			LandmarkObs landmark;
			landmark.x = particle_x + cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y;
			landmark.y = particle_y + sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y;
			
			observations_in_map.push_back(landmark);
		}

		std::vector<LandmarkObs> landmarks;
		for(int i=0; i<map_landmarks.landmark_list.size();i++){
			LandmarkObs landmark;
			landmark.x = map_landmarks.landmark_list[i].x_f;
			landmark.y = map_landmarks.landmark_list[i].y_f;
			landmark.id = i;
			if(dist(particle_x,particle_y,landmark.x, landmark.y) <= sensor_range)
				landmarks.push_back(landmark);
		}
		
		dataAssociation(landmarks,observations_in_map);
	
		double weight = 1.0;
		for(int i=0;i<observations.size();i++){
			double mu_x, mu_y;
			mu_x = landmarks[observations[i].id].x;
			mu_y = landmarks[observations[i].id].y;
			weight *= gaussian(observations[i].x,observations[i].y,mu_x,mu_y,std_landmark[0],std_landmark[1]);
		}

		weights.push_back(weight);
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	std::discrete_distribution<double> d(weights.begin(),weights.end());
	vector<Particle> old_particles = particles;
	for(int i=0; i<num_particles; i++){
		particles[i] = old_particles[d(gen)];
	}	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
