using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using Rect = OpenCvSharp.Rect;

namespace SatGS.ObjectDetection
{
    public class ObjectDetector
    {
        private static ObjectDetector instance;

        public static ObjectDetector Instance()
        {
            if (instance == null)
            {
                instance = new ObjectDetector();
            }
            return instance;
        }

        private ObjectDetector()
        {

        }

        public bool DetectContourOfRedObjects(string imagePath, out Rect? contourRect, out BitmapSource bitmapSource)
        {
            return DetectContourOfRedObjects(new Mat(imagePath), out contourRect, out bitmapSource);
        }

        public bool DetectContourOfRedObjects(Mat image, out Rect? contourRect, out BitmapSource bitmapSource)
        {
            var ranged_image = new Mat();
            var lower = InputArray.Create(new[] { 0, 0, 100 });
            var upper = InputArray.Create(new[] { 60, 60, 255 });
            Cv2.InRange(image, lower, upper, ranged_image);

            Cv2.ImShow("aa",ranged_image);

            Cv2.FindContours(ranged_image, out var contours, out _, RetrievalModes.Tree, ContourApproximationModes.ApproxTC89KCOS);

            Rect? finalRect = null;

            foreach (var contour in contours)
            {
                var rect = Cv2.BoundingRect(contour);
                if (image.Size() == rect.Size) continue;

                if (finalRect == null)
                {
                    finalRect = rect;
                }
                else
                {
                    var tmp = finalRect.Value;
                    tmp |= rect;
                    finalRect = tmp;
                }
            }

            if(finalRect == null)
            {
                contourRect = null;
                bitmapSource = image.ToBitmapSource();

                return false;
            }
            else
            {
                contourRect = finalRect;
                Cv2.Rectangle(image, finalRect.Value, Scalar.Green, 2, LineTypes.AntiAlias);
                bitmapSource = image.ToBitmapSource();

                return true;
            }
        }
    }
}
