/* file: LibUtils.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @brief Intel(R) oneAPI Data Analytics Library package
 */
package com.intel.daal.utils;

import java.io.*;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @ingroup libraryUtilities
 * @{
 */
/**
 * <a name="DAAL-CLASS-LIBUTILS"></a>
 */
public final class LibUtils{
    private static final String LIBRARY_PATH_IN_JAR = "/lib";
    private final static String DAALLIB      = "JavaAPI";
    private final static String TBBLIB       = "tbb";
    private final static String TBBMALLOCLIB = "tbbmalloc";

    private final static String subDir = "daal_" + new Date().getTime();

    private static final Logger logger = Logger.getLogger(LibUtils.class.getName());
    private static final Level logLevel = Level.FINE;

    /**
     * Load JavaAPI DAAL lib and TBB libs
     */
    public static void loadLibrary()
    {
        try {
            logger.log(logLevel, "Loading library " + DAALLIB + " as file.");
            System.loadLibrary(DAALLIB);
            logger.log(logLevel, "DONE: Loading library " + DAALLIB + " as file.");
            return;
        }
        catch (UnsatisfiedLinkError e) {
            logger.log(logLevel, "Can`t find library " + DAALLIB + " in java.library.path.");
        }

        try {
            loadFromJar(subDir, TBBLIB);
            loadFromJar(subDir, TBBMALLOCLIB);
            loadFromJar(subDir, DAALLIB);
            return;
        }
        catch (Throwable e) {
            logger.log(logLevel, "Error: Can`t load library as resource.");
        }
    }

    /**
     * Load lib as resource
     * @param path   sub folder (in temporary folder) name
     * @param name   library name
     */
    private static void loadFromJar(String path, String name) throws IOException
    {
        String FullName = createLibraryFileName(name);

        File fileOut = createTempFile(path, FullName);
        if (fileOut == null) {
            logger.log(logLevel, "DONE: Loading library as resource.");
            return;
        }

        InputStream streamIn = LibUtils.class.getResourceAsStream(LIBRARY_PATH_IN_JAR + "/" + FullName);
        if (streamIn == null)
        {
            throw new IOException("Error: No resource found.");
        }

        try(OutputStream streamOut = new FileOutputStream(fileOut))
        {
            logger.log(logLevel, "Writing resource to temp file.");

            byte[] buffer = new byte[32768];
            while (true)
            {
                int read = streamIn.read(buffer);
                if (read < 0)
                {
                    break;
                }
                streamOut.write(buffer, 0, read);
            }

            streamOut.flush();
        }
        catch (IOException e)
        {
            throw new IOException("Error:  I/O error occurs from/to temp file.");
        }
        finally
        {
            streamIn.close();
        }

        System.load(fileOut.toString());
        logger.log(logLevel, "DONE: Loading library as resource.");
    }

    /**
     * Construct library file name
     * @param name   library name
     *
     * @return constructed file name
     */
    public static String createLibraryFileName(String name) throws IOException
    {
        String fullName;

        String OSname = System.getProperty("os.name");
        OSname = OSname.toLowerCase();

        if (OSname.startsWith("windows")) {
            fullName = name + ".dll";
            return fullName;
        }

        if (OSname.startsWith("linux")) {
            if (name.contains("tbb")) {
                fullName = "lib" + name + ".so.12";
            }
            else {
                fullName = "lib" + name + ".so";
            }
            return fullName;
        }

        if (OSname.startsWith("mac os")) {
            fullName = "lib" + name + ".dylib";
            return fullName;
        }

        throw new IOException("Error: Unknown OS " + OSname );
    }

    /**
     * Create temporary file
     * @param name   library name
     * @param tempSubDirName   sub folder (in temporary folder) name
     *
     * @return temporary file handler. null if file exist already.
     */
    private static File createTempFile(String tempSubDirName, String name) throws IOException
    {
        File tempSubDirectory = new File(System.getProperty("java.io.tmpdir") + "/" + tempSubDirName + LIBRARY_PATH_IN_JAR);

        if (!tempSubDirectory.exists())
        {
            boolean createdDirectory = tempSubDirectory.mkdirs();
            if (!createdDirectory)
            {
                throw new IOException("Error: Can`t create folder for temp file.");
            }
        }

        String tempFileName = tempSubDirectory + "/" + name;
        File tempFile = new File(tempFileName);

        if (tempFile == null)
        {
            throw new IOException("Error: Can`t create temp file.");
        }

        if (tempFile.exists())
        {
            return null;
        }

        return tempFile;
    }

}
/** @} */
