use opencv::{
    core::{no_array, Mat, Point, CV_8UC1, CV_PI, LINE_8},
    highgui::{destroy_all_windows, imshow, wait_key},
    imgcodecs::imread,
    imgproc::{canny, circle, hough_lines_p, line},
};

use opencv::{
    core::Scalar,
    imgproc::{cvt_color, good_features_to_track, COLOR_BGR2GRAY},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfVec4i},
};
use std::error::Error;
fn main() -> Result<(), Box<dyn Error>> {
    let mut img = imread("dashcam-road-rage.jpg", -1)?;
    let mut gray: Mat = Mat::default()?;

    cvt_color(&img, &mut gray, COLOR_BGR2GRAY, 1)?;
    let mut corners = VectorOfPoint2f::new();
    good_features_to_track(
        &gray,
        &mut corners,
        25,
        0.01,
        10.0,
        &no_array()?,
        10,
        false,
        0.04,
    )?;
    for p in corners.iter() {
        circle(
            &mut img,
            p.to().unwrap(),
            3,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
            -1,
            LINE_8,
            0,
        )?;
    }
    let mut lines = VectorOfVec4i::new();
    let mut gray_16 = Mat::default()?;

    gray.convert_to(&mut gray_16, CV_8UC1, 1.0, 0.0)?;
    let mut canny_img = Mat::default()?;
    canny(&mut gray_16, &mut canny_img, 50.0, 100.0, 4, true).unwrap();

    hough_lines_p(&canny_img, &mut lines, 1.0, CV_PI / 180.0, 10, 10., 1.)?;
    for l in lines {
        let point1 = Point::new(l[0], l[1]);
        let point2 = Point::new(l[2], l[3]);
        line(
            &mut img,
            point1,
            point2,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            1,
            8,
            0,
        )?;
    }
    imshow("image", &img)?;
    wait_key(0)?;
    destroy_all_windows()?;

    Ok(())
}
